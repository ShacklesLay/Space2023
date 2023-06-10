import torch
import jsonlines
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

import config
import numpy as np
from torch.utils.data import DataLoader


class Generator_Dataset(Dataset):
    def __init__(self, file_path, config, mode, tokenizer):
        self.mode = mode
        self.device = config.device
        self.tokenizer = tokenizer
        self.dataset = self.preprocess(file_path)

    def preprocess(self, file_path):
        data = []
        if self.mode == 'train':
            with open(file_path, 'r') as f:
                for item in jsonlines.Reader(f):
                    tokens = self.tokenizer.convert_tokens_to_ids(list(item['context']))
                    corefs = item['corefs']
                    for triple in item['outputs']:
                        con_context = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(list(triple[0]['text'])) + \
                                      [self.tokenizer.sep_token_id] + tokens
                        output = [self.tokenizer.cls_token_id]
                        for position, element in enumerate(triple[1:18], 1):

                            if element is None:
                                continue
                            if type(element) == str:
                                continue
                            if type(element) == dict:
                                output.append(self.tokenizer.convert_tokens_to_ids('P' + str(position)))
                                output.extend([self.tokenizer.convert_tokens_to_ids(char) for char in element['text']])
                        output.append(self.tokenizer.convert_tokens_to_ids('[SEP]'))
                        data.append([con_context, output, corefs])

        elif self.mode == 'dev':
            with open(file_path, 'r') as f:
                for item in jsonlines.Reader(f):
                    tokens = self.tokenizer.convert_tokens_to_ids(list(item['context']))
                    corefs = item['corefs']
                    outputs = []
                    for triple in item['outputs']:
                        golden = [None for _ in range(18)]
                        for position, element in enumerate(triple):
                            if element is None:
                                continue
                            elif type(element) in [dict, str]:
                                golden[position] = element
                        outputs.append(golden)
                    data.append([tokens, outputs, corefs])

        elif self.mode == 'test':
            with open(file_path, 'r') as f:
                for item in jsonlines.Reader(f):
                    tokens = self.tokenizer.convert_tokens_to_ids(list(item['context']))
                    data.append([tokens])

        return data

    def __getitem__(self, item):
        tokens = self.dataset[item][0]
        if self.mode == 'train':
            outputs = self.dataset[item][1]
            return [tokens, outputs]
        elif self.mode == 'dev':
            outputs = self.dataset[item][1]
            corefs = self.dataset[item][2]
            return [tokens, outputs, corefs]
        elif self.mode == 'test':
            return [tokens]

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        context = [x[0] for x in batch]
        batch_context = pad_sequence([torch.from_numpy(np.array(c)) for c in context],
                                     batch_first=True,
                                     padding_value=self.tokenizer.pad_token_id)
        batch_context = torch.as_tensor(batch_context, dtype=torch.long).to(self.device)
        if self.mode == 'test':
            return [batch_context]
        elif self.mode == 'train':
            outputs = [x[1] for x in batch]
            outputs = pad_sequence([torch.from_numpy(np.array(o)) for o in outputs],
                                   batch_first=True,
                                   padding_value=self.tokenizer.pad_token_id)
            outputs = torch.as_tensor(outputs, dtype=torch.long).to(self.device)
            return [batch_context, outputs]
        else:
            outputs = [x[1] for x in batch]
            corefs = [x[2] for x in batch]
            return [batch_context, outputs, corefs]


if __name__ == '__main__':
    special_token_dicts = {'additional_special_tokens': ['P' + str(index) for index in range(1, 18)]}
    tokenizer = BertTokenizer.from_pretrained(config.generator_dir)
    tokenizer.add_special_tokens(special_tokens_dict=special_token_dicts)
    dev_dataset = Generator_Dataset(config.test_dir, config, 'test', tokenizer)
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False, collate_fn=dev_dataset.collate_fn)

    for idx, batch_sample in enumerate(dev_loader):
        if idx == 1:
            break
        context = batch_sample
        print(context)
        # print(context)
        # print(label)
        # print([tokenizer.convert_ids_to_tokens(c) for c in context])
        # print([tokenizer.convert_ids_to_tokens(l) for l in label])


