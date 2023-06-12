import jsonlines
import numpy as np
import torch

import config
import logging
from data_loader import SpaceDataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import  DebertaReaderfortask
from utils import set_logger


def generate():
    set_logger()
    dataset = SpaceDataset(config.test_dir, config, mode='test')
    logging.info('Dataset build!')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)
    logging.info('Dataloader build!')
    tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_cased=config.bert_cased)
    modelfortask = DebertaReaderfortask.from_pretrained(config.model_dir) 
    modelfortask.to(config.device)    
    logging.info('Load the model from {}'.format(config.model_dir))    
    logging.info('Test Beginning!')
    modelfortask.eval()

    result = []
    with torch.no_grad():
        for idx, batch_sample in enumerate(dataloader):

            result.append([])
            batch_data = batch_sample[0]
            batch_mask = batch_data.gt(0)
            
            CLS = torch.tensor(tokenizer.cls_token_id).expand(1, 1).to(config.device)
            batch_data = torch.cat((CLS, batch_data), dim=-1)
            batch_mask = batch_data.gt(0)
            outputs2 = modelfortask(batch_data, batch_mask)            
            result[idx].append([torch.argmax(outputs2['logits'][:, :, i], dim=-1).item() for i in range(12)])
            for ind in range(0, 12, 2):
                if result[idx][0][ind] > result[idx][0][ind + 1]:
                    result[idx][0][ind], result[idx][0][ind + 1] = result[idx][0][ind + 1], result[idx][0][ind]

    return result


if __name__ == '__main__':
    _ = generate()
    with open(config.test_dir, 'r') as fr:
        items = []
        for idx, item in enumerate(jsonlines.Reader(fr)):
            result = _[idx]
            reason = []
            qid, context, reasons = item['qid'], item['context'], []
            result2=result[0]
            frag2 = []
            if result2[0] != 0 and result2[1] != 0:
                frag2.append({'role': 'S1', 'idxes': [v for v in range(result2[0] - 1, result2[1])]})
            if result2[2] != 0 and result2[3] != 0:
                frag2.append({'role': 'P1', 'idxes': [v for v in range(result2[2] - 1, result2[3])]})
            if result2[4] != 0 and result2[5] != 0:
                frag2.append({'role': 'E1', 'idxes': [v for v in range(result2[4] - 1, result2[5])]})
            if result2[6] != 0 and result2[7] != 0:
                frag2.append({'role': 'S2', 'idxes': [v for v in range(result2[6] - 1, result2[7])]})
            if result2[8] != 0 and result2[9] != 0:
                frag2.append({'role': 'P2', 'idxes': [v for v in range(result2[8] - 1, result2[9])]})
            if result2[10] != 0 and result2[11] != 0:
                frag2.append({'role': 'E2', 'idxes': [v for v in range(result2[10] - 1, result2[11])]})
            reason.append(frag2)
            items.append({'qid': qid, 'context': context, "results": reason})

    with jsonlines.open(config.prediction_dir, 'w') as fw:
        fw.write_all(items)
