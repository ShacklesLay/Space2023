import jsonlines
import torch
import config
import logging
from utils import set_logger
from torch.utils.data import DataLoader
from transformers import BertTokenizer, DebertaForTokenClassification
from modeling_cpt import CPTForConditionalGeneration
from gen_dataloader import Generator_Dataset
from config import generator_params as params


def generate():
    set_logger(config.generation_log_dir, config.generation_model_dir)
    logging.info('device: {}'.format(config.device))

    gen_tokenizer = BertTokenizer.from_pretrained(config.generator_dir)
    special_token_dicts = {'additional_special_tokens': ['P' + str(index) for index in range(1, 18)]}
    gen_tokenizer.add_special_tokens(special_tokens_dict=special_token_dicts)

    sel_tokenizer = BertTokenizer.from_pretrained(config.selector_dir)

    dataset = Generator_Dataset(config.test_dir, config, mode='test', tokenizer=gen_tokenizer)
    # dataset = Generator_Dataset(config.test_dir, config, mode='test', tokenizer=gen_tokenizer)
    logging.info('Dataset Build!')

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)
    logging.info('Dataloader Build!')

    gen_model = CPTForConditionalGeneration.from_pretrained(config.generator_model_dir)
    gen_model.resize_token_embeddings(len(gen_tokenizer))
    gen_model.to(config.device)
    logging.info('Load Generator From {}'.format(config.generator_model_dir))

    sel_model = DebertaForTokenClassification.from_pretrained(config.selector_model_dir)
    sel_model.to(config.device)
    logging.info('Load Selector From {}'.format(config.selector_model_dir))

    logging.info('Starting Testing')

    gen_model.eval()
    sel_model.eval()
    with torch.no_grad():
        generated_result = []
        for batch_sample in dataloader:
            batch_data = batch_sample[0]
            batch_mask = batch_data.gt(0)
            data = batch_data[0]
            data = gen_tokenizer.convert_ids_to_tokens([id for id in data])
            data = sel_tokenizer.convert_tokens_to_ids([token for token in data])
            batch_data = torch.as_tensor([data], dtype=torch.long).to(config.device)
            outputs = sel_model(batch_data, batch_mask)
            pred = torch.argmax(outputs['logits'], dim=-1)[0].cpu().numpy().tolist()
            main_body_set = []
            for pos, target in enumerate(pred):
                if (pos == 0 or pred[pos - 1] == 0) and pred[pos] == 1:
                    start_position = pos
                if ((pos == len(pred) - 1) or (pred[pos + 1] == 0)) and pred[pos] == 1:
                    end_position = pos
                    main_body_set.append([start_position, end_position])
            batch_data = batch_data[0]
            batch_data = sel_tokenizer.convert_ids_to_tokens([id for id in batch_data])
            batch_data = gen_tokenizer.convert_tokens_to_ids([token for token in batch_data])
            prediction = []
            for main_body in main_body_set:
                con_context = [gen_tokenizer.cls_token_id] + \
                              batch_data[main_body[0]: main_body[1] + 1] + \
                              [gen_tokenizer.sep_token_id] + \
                              batch_data
                con_context = torch.as_tensor([con_context], dtype=torch.long).to(torch.device('cuda'))
                rest_elements = gen_model.generate(con_context, num_beams=params['num_beams'],
                                                   do_sample=params['do_sample'], max_length=300)
                rest_elements = gen_tokenizer.batch_decode(rest_elements)
                start_element = gen_tokenizer.convert_ids_to_tokens(
                    token for token in batch_data[main_body[0]:main_body[1] + 1])
                rest_elements = rest_elements[0].split()[2:-1]
                start_element = ''.join(start_element)
                dest_tuple = [None for _ in range(18)]
                for e in rest_elements:
                    if e in ['P' + str(i) for i in range(1, 18)]:
                        cur = int(e[1:])
                    else:
                        if dest_tuple[cur] is None:
                            dest_tuple[cur] = []
                            dest_tuple[cur].append(e)
                        else:
                            dest_tuple[cur].append(e)
                batch_text = ''.join(gen_tokenizer.convert_ids_to_tokens(id for id in batch_data))
                dest_tuple[0] = {'text': start_element,
                                 'idxes': [idx for idx in range(main_body[0], main_body[1] + 1)]}
                for pos, element in enumerate(dest_tuple):
                    if pos == 0:
                        continue
                    if element is None:
                        continue
                    found = batch_text.find(''.join(element))
                    if found != -1:
                        dest_tuple[pos] = {'text': ''.join(element),
                                           'idxes': [i for i in range(found, found + len(''.join(element)))]}
                    else:
                        last_position_list = []
                        for char in element:
                            idx = batch_text.find(char, last_position_list[-1]) if last_position_list else batch_text.find(char)
                            if idx != -1:
                                last_position_list.append(idx)
                        dest_tuple[pos] = {'text': ''.join(element), 'idxes': last_position_list}
                prediction.append(dest_tuple)
            generated_result.append(prediction)
    return generated_result


if __name__ == '__main__':
    result = generate()
    with open(config.test_dir, 'r') as fr:
        items = []
        for idx, item in enumerate(jsonlines.Reader(fr)):
            qid, context = item['qid'], item['context']
            items.append({'qid': qid, 'context': context, 'outputs': result[idx]})

    with jsonlines.open(config.prediction_dir, 'w') as fw:
        fw.write_all(items)