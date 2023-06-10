import torch

import config
import logging
from torch.optim import AdamW
from config import generator_params
from gen_dataloader import Generator_Dataset
from torch.utils.data import DataLoader
from utils import set_logger, set_seed
from transformers import BertTokenizer, DebertaForTokenClassification
from transformers.optimization import get_cosine_schedule_with_warmup
from modeling_cpt import CPTForConditionalGeneration
from train_generator import train


def run():
    set_logger(config.generator_log_dir, config.generator_model_dir)
    logging.info('device {}'.format(config.device))
    set_seed(generator_params['seed'])

    gen_tokenizer = BertTokenizer.from_pretrained(config.generator_dir)
    sel_tokenizer = BertTokenizer.from_pretrained(config.selector_dir)
    special_token_dicts = {'additional_special_tokens': ['P' + str(index) for index in range(1, 18)]}
    gen_tokenizer.add_special_tokens(special_tokens_dict=special_token_dicts)

    train_dataset = Generator_Dataset(config.train_dir, config, mode='train', tokenizer=gen_tokenizer)
    dev_dataset = Generator_Dataset(config.dev_dir, config, mode='dev', tokenizer=gen_tokenizer)
    logging.info('Dataset Build!')
    train_loader = DataLoader(train_dataset, batch_size=generator_params['batch_size'],
                              shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=1,
                            shuffle=False, collate_fn=dev_dataset.collate_fn)
    logging.info('Dataloader Build!')
    gen_model = CPTForConditionalGeneration.from_pretrained(config.generator_dir)
    gen_model.resize_token_embeddings(len(gen_tokenizer))
    gen_model.to(config.device)
    logging.info('Load Gen_Model and Tokenizer From {}'.format(config.generator_dir))

    sel_model = DebertaForTokenClassification.from_pretrained(config.selector_model_dir)
    sel_model.to(config.device)
    logging.info('Load Sel_Model and Tokenizer From {}'.format(config.selector_model_dir))

    train_steps_per_epoch = len(train_dataset) // generator_params['batch_size']
    optimizer = AdamW(gen_model.parameters(), lr=generator_params['lr'])
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=train_steps_per_epoch,
                                                num_training_steps=train_steps_per_epoch * generator_params['epoch'])
    logging.info('Starting Training')
    train(train_loader, dev_loader, sel_model,  gen_model, optimizer, scheduler,
          config.generator_model_dir, sel_tokenizer, gen_tokenizer)


if __name__ == '__main__':
    run()