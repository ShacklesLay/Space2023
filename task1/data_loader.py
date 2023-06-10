import torch
import jsonlines
import numpy as np
from transformers import BertTokenizer, DebertaTokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import config
from torch.utils.data import DataLoader

#          train   dev
# subtask1 1258    201
# subtask2 910     112
# subtask3 3750    503


class SpaceDataset(Dataset):

    def __init__(self, file_path, config, mode: str, subtask=None):
        self.mode = mode
        # in ['train'/'dev'/'test']
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=config.bert_cased)
        self.device = config.device
        self.subtask = config.subtask
        self.dataset = self.preprocess(file_path)


    def preprocess(self, file_path):

        if self.subtask == 'subtask1':
            data = []
            with open(file_path, "r") as fr:
                for item in jsonlines.Reader(fr):
                    text = item['context']
                    tokens = self.tokenizer.convert_tokens_to_ids(list(text))
                    if self.mode == 'train':
                        for reason in reasons:
                            fragment = reason['fragments']
                            role1, role2 = fragment[0], fragment[1]
                            labels.extend([role1['idxes'][0], role1['idxes'][-1]])
                            labels.extend([role2['idxes'][0], role2['idxes'][-1]])
                        data.append([tokens, labels])
                    elif self.mode == 'test':
                        data.append([tokens])
                    elif self.mode == 'dev':
                        for reason in reasons:
                            fragment = reason['fragments']
                            role1, role2 = fragment[0], fragment[1]
                            labels.append(role1['idxes'])
                            labels.append(role2['idxes'])
                        data.append([tokens, labels])
            return data

        elif self.subtask == 'subtask2':
            data = []
            with open(file_path, 'r') as fr:
                for item in jsonlines.Reader(fr):
                    text, reasons, labels = item['context'], item['results'], []
                    tokens = self.tokenizer.convert_tokens_to_ids(list(text))
                    CLS = [self.tokenizer.cls_token_id]
                    if self.mode == 'test':
                        data.append([CLS + tokens])
                    elif self.mode == 'train':
                        for reason in reasons:
                            answer = [0] * 12
                            fragment = reason
                            for element in fragment:
                                element['idxes'].sort()
                                start_position = element['idxes'][0] + 1
                                end_position = element['idxes'][-1] + 1
                                if element['role'] == 'S1':
                                    answer[0], answer[1] = start_position, end_position
                                elif element['role'] == 'P1':
                                    answer[2], answer[3] = start_position, end_position
                                elif element['role'] == 'E1':
                                    answer[4], answer[5] = start_position, end_position
                                elif element['role'] == 'S2':
                                    answer[6], answer[7] = start_position, end_position
                                elif element['role'] == 'P2':
                                    answer[8], answer[9] = start_position, end_position
                                elif element['role'] == 'E2':
                                    answer[10], answer[11] = start_position, end_position
                            labels.extend(answer)
                        data.append([CLS + tokens, labels])
                    elif self.mode == 'dev':
                        mapping = {'S1': 0, 'P1': 1, 'E1': 2, 'S2': 3, 'P2': 4, 'E2': 5}
                        for reason in reasons:
                            fragment = reason
                            answer = [[]] * 6
                            for element in fragment:
                                idxes = [item + 1 for item in element['idxes']]
                                answer[mapping[element['role']]] = idxes
                            labels.extend(answer)
                        data.append([CLS + tokens, labels])
            return data

        elif self.subtask == 'subtask3':
            # [CLS] w1 w2 ... wn
            data = []
            with open(file_path, 'r') as fr:
                for item in jsonlines.Reader(fr):
                    text, reasons, labels = item['context'], item['reasons'], []
                    tokens = self.tokenizer.convert_tokens_to_ids(list(text))
                    CLS = [self.tokenizer.cls_token_id]
                    if self.mode == 'test':
                        data.append([CLS + tokens])
                    elif self.mode == 'train':
                        for reason in reasons:
                            answer = [0] * 6
                            fragment = reason['fragments']
                            for element in fragment:
                                element['idxes'].sort()
                                if element['role'] == 'S':
                                    answer[0], answer[1] = element['idxes'][0] + 1, element['idxes'][-1] + 1
                                elif element['role'] == 'P':
                                    answer[2], answer[3] = element['idxes'][0] + 1, element['idxes'][-1] + 1
                                elif element['role'] == 'E':
                                    answer[4], answer[5] = element['idxes'][0] + 1, element['idxes'][-1] + 1
                            labels.extend(answer)
                        data.append([CLS + tokens, labels])
                    elif self.mode == 'dev':
                        for reason in reasons:
                            fragment = reason['fragments']
                            answer = [[]] * 3
                            for element in fragment:
                                idxes = [item + 1 for item in element['idxes']]
                                if element['role'] == 'S':
                                    answer[0] = idxes
                                elif element['role'] == 'P':
                                    answer[1] = idxes
                                elif element['role'] == 'E':
                                    answer[2] = idxes
                            labels.extend(answer)
                        data.append([CLS + tokens, labels])
            return data

    def __getitem__(self, idx):
        tokens = self.dataset[idx][0]
        if self.mode == 'test':
            return [tokens]
        else:
            label = self.dataset[idx][1]
            return [tokens, label]

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        if self.subtask == 'subtask1':
            sentence = [x[0] for x in batch]
            batch_data = pad_sequence([torch.from_numpy(np.array(s)) for s in sentence], batch_first=True, padding_value=self.tokenizer.pad_token_id)
            batch_data = torch.as_tensor(batch_data, dtype=torch.long).to(self.device)
            if self.mode == 'test':
                return [batch_data]
            elif self.mode == 'train':
                # setting 1: choose the shorter span
                labels = [x[1] for x in batch]
                for idx, label in enumerate(labels):
                    if len(label) == 8:
                        if (label[1] - label[0] + label[3] - label[2]) < (label[5] - label[4] + label[7] - label[6]):
                            labels[idx] = label[0:4]
                        else:
                            labels[idx] = label[4:8]
                batch_label = torch.as_tensor(labels, dtype=torch.long).to(self.device)
                return [batch_data, batch_label]

                # setting 2: choose all the spans
                # setting 3: separate into two parts
            else:
                 labels = [x[1] for x in batch]
                 return [batch_data, labels]

        elif self.subtask == 'subtask2':
            sentence = [x[0] for x in batch]
            batch_data = pad_sequence([torch.from_numpy(np.array(s)) for s in sentence], batch_first=True,
                                      padding_value=self.tokenizer.pad_token_id)
            batch_data = torch.as_tensor(batch_data, dtype=torch.long).to(self.device)
            if self.mode == 'test':
                return [batch_data]
            elif self.mode == 'train':
                labels = [x[1] for x in batch]
                for idx, label in enumerate(labels):
                    if len(label) > 12:
                        labels[idx] = label[0:12]
                batch_label = torch.as_tensor(labels, dtype=torch.long).to(self.device)
                return [batch_data, batch_label]
            else:
                labels = [x[1] for x in batch]
                return [batch_data, labels]
        elif self.subtask == 'subtask3':
            sentence = [x[0] for x in batch]
            batch_data = pad_sequence([torch.from_numpy(np.array(s)) for s in sentence], batch_first=True, padding_value=self.tokenizer.pad_token_id)
            batch_data = torch.as_tensor(batch_data, dtype=torch.long).to(self.device)
            if self.mode == 'test':
                return [batch_data]
            elif self.mode == 'train':
                labels = [x[1] for x in batch]
                for idx, label in enumerate(labels):
                    if len(label) > 6:
                        labels[idx] = label[0:6]
                batch_label = torch.as_tensor(labels, dtype=torch.long).to(self.device)
                return [batch_data, batch_label]
            else:
                labels = [x[1] for x in batch]
                return [batch_data, labels]

    @staticmethod
    def filter(span: list):

        if span[-1] - span[0] + 1 == len(span):
            return [span[0], span[-1]]
        else:
            segment1, segment2 = [], []
            segment1.append(span[0])
            segment2.append(span[-1])
            for i in range(1, len(span)):
                if span[i] - span[i - 1] == 1:
                    segment1.append(span[i])
                else:
                    break
            for i in range(len(span) - 2, 0, -1):
                if span[i + 1] - span[i] == 1:
                    segment2.append(span[i])
                else:
                    break
            segment1.sort()
            segment2.sort()
            if len(segment1) > len(segment2):
                return [segment1[0], segment1[-1]]
            else:
                return [segment2[0], segment2[-1]]


if __name__ == '__main__':

    train_dataset = SpaceDataset(config.train_dir, config, 'train')
    dev_dataset = SpaceDataset(config.dev_dir, config, 'dev')

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=4, shuffle=False, collate_fn=dev_dataset.collate_fn)

    for idx, sample in enumerate(train_loader):
        if idx == 6:
            break
        batch_data, batch_label = sample
        batch_mask = batch_data.gt(0)
        print(idx, batch_data.shape, batch_label.shape)
        print(batch_label)

    for idx, sample in enumerate(dev_loader):
        if idx == 6:
            break
        batch_data, batch_label = sample
        batch_mask = batch_data.gt(0)
        # print(idx, batch_data.shape, batch_label)
        print(batch_label)

