import torch
import jsonlines
import numpy as np
from transformers import BertTokenizer, DebertaTokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import config
from torch.utils.data import DataLoader

class SpaceDataset(Dataset):

    def __init__(self, file_path, config, mode: str):
        self.mode = mode
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=config.bert_cased)
        self.device = config.device
        self.dataset = self.preprocess(file_path)


    def preprocess(self, file_path):
        data = []
        with open(file_path, 'r') as fr:
            for item in jsonlines.Reader(fr):
                text, reasons, labels = item['context'], item.get('results'), []
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

