import jsonlines
import numpy as np
import torch

import config
import logging
from data_loader import SpaceDataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import DebertaReaderfortask1, DebertaReaderfortask2, DebertaReaderfortask3
from utils import set_logger


def generate():
    set_logger()
    dataset = SpaceDataset(config.dev_dir, config, mode='test')
    logging.info('Dataset build!')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)
    logging.info('Dataloader build!')
    tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_cased=config.bert_cased)
    # modelfortask1 = DebertaReaderfortask1.from_pretrained(config.subtask1_model_dir)
    modelfortask2 = DebertaReaderfortask2.from_pretrained(config.subtask2_model_dir)
    # modelfortask3 = DebertaReaderfortask3.from_pretrained(config.subtask3_model_dir)
    # modelfortask1.to(config.device)
    modelfortask2.to(config.device)
    # modelfortask3.to(config.device)
    # logging.info('Load the model from {}'.format(config.subtask1_model_dir))
    logging.info('Load the model from {}'.format(config.subtask2_model_dir))
    # logging.info('Load the model from {}'.format(config.subtask3_model_dir))
    logging.info('Test Beginning!')

    # modelfortask1.eval()
    modelfortask2.eval()
    # modelfortask3.eval()
    # gap = 100

    result = []
    with torch.no_grad():
        for idx, batch_sample in enumerate(dataloader):

            result.append([])
            batch_data = batch_sample[0]
            batch_mask = batch_data.gt(0)
            # outputs1 = modelfortask1(batch_data, batch_mask)

            CLS = torch.tensor(tokenizer.cls_token_id).expand(1, 1).to(config.device)
            batch_data = torch.cat((CLS, batch_data), dim=-1)
            batch_mask = batch_data.gt(0)
            outputs2 = modelfortask2(batch_data, batch_mask)
            # outputs3 = modelfortask3(batch_data, batch_mask)
            # subtask1
            # text1_S = torch.argmax(outputs1['A_start_logits'], dim=-1).item()
            # text1_E = torch.argmax(outputs1['A_end_logits'], dim=-1).item()
            # text2_S = torch.argmax(outputs1['B_start_logits'], dim=-1).item()
            # text2_E = torch.argmax(outputs1['B_end_logits'], dim=-1).item()
            # if text1_S > text1_E:
            #     text1_S, text1_E = text1_E, text1_S
            # #if text1_E - text1_S > gap:
            # #    text1_S = text1_E - gap
            # if text2_S > text2_E:
            #     text2_S, text2_E = text2_E, text2_S
            # #if text2_E - text2_S > gap:
            # #    text2_S = text2_E - gap
            # result[idx].append([text1_S, text1_E, text2_S, text2_E])

            # subtask2
            result[idx].append([torch.argmax(outputs2['logits'][:, :, i], dim=-1).item() for i in range(12)])
            for ind in range(0, 12, 2):
                if result[idx][0][ind] > result[idx][0][ind + 1]:
                    result[idx][0][ind], result[idx][0][ind + 1] = result[idx][0][ind + 1], result[idx][0][ind]
                #if result[idx][1][ind + 1] - result[idx][1][ind] > gap:
                #    result[idx][1][ind] = result[idx][1][ind + 1] - gap

            # # subtask3
            # S_start = torch.argmax(outputs3['S_start_logits'], dim=-1).item()
            # S_end = torch.argmax(outputs3['S_end_logits'], dim=-1).item()
            # P_start = torch.argmax(outputs3['P_start_logits'], dim=-1).item()
            # P_end = torch.argmax(outputs3['P_end_logits'], dim=-1).item()
            # E_start = torch.argmax(outputs3['E_start_logits'], dim=-1).item()
            # E_end = torch.argmax(outputs3['E_end_logits'], dim=-1).item()
            # if S_start > S_end:
            #     S_start, S_end = S_end, S_start
            # #if S_end - S_start > gap:
            # #    S_start = S_end - gap
            # if P_start > P_end:
            #     P_start, P_end = P_end, P_start
            # #if P_end - P_start > gap:
            # #    P_start = P_end - gap
            # if E_start > E_end:
            #     E_start, E_end = E_end, E_start
            # #if E_end - E_start > gap:
            # #    E_start = E_end - gap
            # result[idx].append([S_start, S_end, P_start, P_end, E_start, E_end])

    return result


if __name__ == '__main__':
    _ = generate()
    with open(config.dev_dir, 'r') as fr:
        items = []
        for idx, item in enumerate(jsonlines.Reader(fr)):
            result = _[idx]
            reason = []
            qid, context, reasons = item['qid'], item['context'], []
            # result1, result2, result3 = result[0], result[1], result[2]
            result2=result[0]

            # frag1 = []
            # frag1.append({'role': 'text1', 'idxes': [v for v in range(result1[0], result1[1] + 1)]})
            # frag1.append({'role': 'text2', 'idxes': [v for v in range(result1[2], result1[3] + 1)]})
            # fragment1 = {'fragments': frag1, 'type': 'A'}
            # reason.append(fragment1)

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

            # frag3 = []
            # if result3[0] != 0 and result3[1] != 0:  # result3[0] + 8
            #     frag3.append({'role': 'S', 'idxes': [v for v in range(result3[0] - 1, result3[1])]})
            # if result3[2] != 0 and result3[3] != 0:
            #     frag3.append({'role': 'P', 'idxes': [v for v in range(result3[2] - 1, result3[3])]})
            # if result3[4] != 0 and result3[5] != 0:
            #     frag3.append({'role': 'E', 'idxes': [v for v in range(result3[4] - 1, result3[5])]})
            # fragment3 = {'fragments': frag3, 'type': 'C'}
            # reason.append(fragment3)

            items.append({'qid': qid, 'context': context, "results": reason})

    with jsonlines.open(config.prediction_dir, 'w') as fw:
        fw.write_all(items)
