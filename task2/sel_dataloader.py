import torch
import jsonlines
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence


class Selector_Dataset(Dataset):
    def __init__(self, file_path, config, mode):
        self.mode = mode
        self.device = config.device
        self.tokenizer = BertTokenizer.from_pretrained(config.selector_dir)
        self.dataset = self.preprocess(file_path)

    def preprocess(self, file_path):
        data = []
        if self.mode in ['train', 'dev']:
            with open(file_path, 'r') as f:
                for _, item in enumerate(jsonlines.Reader(f)):
                    tokens = self.tokenizer.convert_tokens_to_ids(list(item['context']))
                    label = [0 for _ in range(len(tokens))]
                    for triple in item['outputs']:
                        for idx in triple[0]['idxes']:
                            label[idx] = 1
                    data.append([tokens, label])
        else:
            with open(file_path, 'r') as f:
                for _, item in enumerate(jsonlines.Reader(f)):
                    tokens = self.tokenizer.convert_tokens_to_ids(list(item['context']))
                    data.append([tokens])
        return data

    def __getitem__(self, item):
        tokens = self.dataset[item][0]
        if self.mode in ['train', 'dev']:
            label = self.dataset[item][1]
            return [tokens, label]
        else:
            return [tokens]

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        context = [x[0] for x in batch]
        batch_data = pad_sequence([torch.from_numpy(np.array(c)) for c in context],
                                  batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch_data = torch.as_tensor(batch_data, dtype=torch.long).to(self.device)

        if self.mode in ['train', 'dev']:
            label = [x[1] for x in batch]
            batch_label = pad_sequence([torch.from_numpy(np.array(l)) for l in label],
                                  batch_first=True, padding_value=self.tokenizer.pad_token_id)
            batch_label = torch.as_tensor(batch_label, dtype=torch.long).to(self.device)
            return [batch_data, batch_label]
        else:
            return [batch_data]

