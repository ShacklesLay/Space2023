import config
import logging

from utils import set_logger, set_seed
from data_loader import SpaceDataset
from torch.utils.data import DataLoader
from model import DebertaReaderfortask1, DebertaReaderfortask2, DebertaReaderfortask3
from model import DebertaReaderfortask2MLP
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from train import train


def run():
    set_logger()
    logging.info('device: {}'.format(config.device))
    set_seed(config.seed)
    train_dataset = SpaceDataset(config.train_dir, config, mode='train')
    dev_dataset = SpaceDataset(config.dev_dir, config, mode='dev')
    logging.info('Dataset Bulid!')

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=config.shuffle, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=config.shuffle, collate_fn=dev_dataset.collate_fn)

    logging.info('Get DataLoader!')
    if config.subtask == 'subtask1':
        logging.info('running program for subtask1!')
        model = DebertaReaderfortask1.from_pretrained(config.bert_model)
    elif config.subtask == 'subtask2':
        logging.info('running program for subtask2!')
        model = DebertaReaderfortask2.from_pretrained(config.bert_model)
    elif config.subtask == 'subtask3':
        logging.info('running program for subtask3!')
        model = DebertaReaderfortask3.from_pretrained(config.bert_model)

    model.to(config.device)
    logging.info('Load Model Form {}'.format(config.bert_model))

    train_steps_per_epoch = len(train_dataset) // config.batch_size
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                            num_warmup_steps=train_steps_per_epoch,
                                            num_training_steps=train_steps_per_epoch * config.epoch)
    logging.info('Starting Training')
    train(train_loader, dev_loader, model, optimizer, scheduler, config.model_dir)


if __name__ == '__main__':
    run()
