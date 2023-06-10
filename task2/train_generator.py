from tqdm import tqdm

import torch
import logging
import numpy as np
from scipy import optimize

from config import generator_params as params
import torch.nn as nn


def train_epoch(train_loader, model, optimizer, scheduler):
    model.train()
    train_loss = 0.0
    for _, batch_sample in enumerate(tqdm(train_loader)):
        batch_data, batch_label = batch_sample
        batch_mask = batch_data.gt(0)
        outputs = model(input_ids=batch_data,
                        attention_mask=batch_mask,
                        labels=batch_label)
        loss = outputs['loss']
        train_loss += loss.item()
        model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=params['clip_grad'])
        optimizer.step()
        scheduler.step()
    train_loss = float(train_loss) / len(train_loader)
    return train_loss


def evaluate(dev_loader, sel_model, gen_model, sel_tokenizer, gen_tokenizer):
    sel_model.eval()
    gen_model.eval()
    with torch.no_grad():
        sum_p, sum_r, sum_f = 0.0, 0.0, 0.0
        generated_result = []
        for batch_sample in tqdm(dev_loader):
            batch_data, batch_label, coref = batch_sample
            batch_mask = batch_data.gt(0)
            batch_data = batch_data[0]
            batch_data = gen_tokenizer.convert_ids_to_tokens([id for id in batch_data])
            batch_data = sel_tokenizer.convert_tokens_to_ids([token for token in batch_data])
            batch_data = torch.as_tensor([batch_data], dtype=torch.long).to(torch.device('cuda'))
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
            pred = []
            for main_body in main_body_set:
                con_context = [gen_tokenizer.cls_token_id] + \
                              batch_data[main_body[0]: main_body[1] + 1] + \
                              [gen_tokenizer.sep_token_id] + \
                              batch_data
                con_context = torch.as_tensor([con_context], dtype=torch.long).to(torch.device('cuda'))
                rest_elements = gen_model.generate(con_context, num_beams=params['num_beams'], do_sample=params['do_sample'], max_length=300)
                rest_elements = gen_tokenizer.batch_decode(rest_elements)
                start_element = gen_tokenizer.convert_ids_to_tokens(token for token in batch_data[main_body[0]:main_body[1]+1])
                rest_elements = rest_elements[0].split()[2:-1]
                start_element = ''.join(start_element)
                dest_tuple = [None for _ in range(18)]
                for e in rest_elements:
                    if e in ['P'+str(i) for i in range(1, 18)]:
                        cur = int(e[1:])
                    else:
                        if dest_tuple[cur] is None:
                            dest_tuple[cur] = []
                            dest_tuple[cur].append(e)
                        elif type(dest_tuple[cur]) == list:
                            dest_tuple[cur].append(e)
                batch_text = ''.join(gen_tokenizer.convert_ids_to_tokens(id for id in batch_data))
                dest_tuple[0] = {'text': [start_element], 'idxes': [idx for idx in range(main_body[0], main_body[1] + 1)]}
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
                            idx = batch_text.find(char,
                                                  last_position_list[-1]) if last_position_list else batch_text.find(
                                char)
                            if idx != -1:
                                last_position_list.append(idx)
                        dest_tuple[pos] = {'text': ''.join(element), 'idxes': last_position_list}
                pred.append(dest_tuple)
            # generated_result.append(pred)
            p, r, f = get_metric(pred, batch_label[0], coref[0])
            sum_p += p
            sum_r += r
            sum_f += f

    return {
        'precision': sum_p/len(dev_loader),
        'recall': sum_r/len(dev_loader),
        'f1': sum_f/len(dev_loader)
    }


def intersection_and_union(input, target):
    _input, _target = set(input), set(target)
    intersection = _input & _target
    union = _input | _target
    return len(intersection), len(union)


def cal_similarity(golden_tuple, predicted_tuple, corefs):
    if (len(golden_tuple) != len(predicted_tuple)):
        return 0

    non_null_pair = 0
    total_score = 0.0
    for i, (g_element, p_element) in enumerate(zip(golden_tuple, predicted_tuple)):
        if (g_element is None) and (p_element is None):
            continue
        non_null_pair += 1
        if (g_element is None) or (p_element is None):
            element_sim_score = 0
        else:
            if (isinstance(g_element, str)):  # 标签类元素
                if not (isinstance(p_element, str)):
                    element_sim_score = 0.0
                elif (g_element != p_element):
                    element_sim_score = 0.0
                else:
                    element_sim_score = 1.0
            else:
                p_idx, g_idx = p_element['idxes'], g_element['idxes']
                p_text, g_text = p_element['text'], g_element['text']

                n_inter, n_union = intersection_and_union(p_text, g_text)
                element_sim_score = n_inter / n_union

                if ((i == 0) or (i == 1)):
                    n_inter, n_union = intersection_and_union(p_idx, g_idx)

                    element_sim_score = n_inter / n_union
                    g_idx_set = set(g_idx)
                    for key in corefs:
                        key_idx_set = set(eval(key))
                        if (key_idx_set.issubset(g_idx_set)):
                            diff_set = g_idx_set - key_idx_set
                            for c in corefs[key]:
                                corefed_g_idx = set(c['idxes']) | diff_set
                                n_inter, n_union = intersection_and_union(p_idx, corefed_g_idx)
                                element_sim_score = max(element_sim_score, n_inter / n_union)

        if ((i == 0) or (i == 1)) and (element_sim_score == 0):
               return 0
        total_score += element_sim_score
    return total_score / non_null_pair


def KM_algorithm(pair_scores):
    row_ind, col_ind = optimize.linear_sum_assignment(-pair_scores)
    max_score = pair_scores[row_ind, col_ind].sum()
    return max_score


def get_metric(pred, label, corefs):
    N, M = len(pred), len(label)
    pair_scores = np.zeros((N, M))
    coref_dict = {}
    for coref_set in corefs:
        for coref_element in coref_set:
            idx_str = str(coref_element['idxes'])
            if idx_str not in coref_dict:
                coref_dict[idx_str] = coref_set
    for i in range(N):
        for j in range(M):
            pair_scores[i][j] = cal_similarity(pred[i], label[j], coref_dict)
        max_bipartite_score = KM_algorithm(pair_scores)
        precision = max_bipartite_score / N
        recall = max_bipartite_score / M
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def train(train_loader, dev_loader, sel_model, gen_model, optimizer, scheduler, model_dir, sel_tokenizer, gen_tokenizer):
    best_f1 = 0.0
    patience_counter = 0
    for epoch in range(1, params['epoch'] + 1):
        train_loss = train_epoch(train_loader, gen_model, optimizer, scheduler)
        metrics = evaluate(dev_loader, sel_model, gen_model, sel_tokenizer, gen_tokenizer)
        precision, recall, f1 = metrics['precision'], metrics['recall'], metrics['f1']
        logging.info('Epoch {}, Train_loss: {:.6f} f1:{:.6f}, precision:{:.6f}, recall:{:.6f}'
                     .format(epoch, train_loss, f1, precision, recall))
        if f1 > best_f1:
            gen_model.save_pretrained(model_dir)
            logging.info('Save best model!')
            if f1 - best_f1 < params['patience']:
                patience_counter += 1
            else:
                patience_counter = 0
            best_f1 = f1
        else:
            patience_counter += 1
            if (patience_counter >= params['patience_num'] and epoch > params['min_epoch_num']) or epoch == params['epoch']:
                logging.info('Best val accuracy: {}'.format(best_f1))
                break
            logging.info('Training Finished!')