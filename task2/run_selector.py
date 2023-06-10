import config
import logging
from torch.optim import AdamW
from config import selector_params
from sel_dataloader import Selector_Dataset
from torch.utils.data import DataLoader
from utils import set_logger, set_seed
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import DebertaForTokenClassification
from train_selector import train


def run():
    set_logger(config.selector_log_dir, config.selector_model_dir)
    logging.info('device: {}'.format(config.device))
    set_seed(selector_params['seed'])
    train_dataset = Selector_Dataset(config.train_dir, config, mode='train')
    dev_dataset = Selector_Dataset(config.dev_dir, config, mode='dev')
    logging.info('Dataset Build!')

    train_loader = DataLoader(train_dataset, batch_size=selector_params['batch_size'],
                              shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=selector_params['batch_size'],
                            shuffle=False, collate_fn=dev_dataset.collate_fn)

    logging.info('Dataloader Build!')
    model = DebertaForTokenClassification.from_pretrained(config.selector_dir, num_labels=2)
    model.to(config.device)
    logging.info('Load Model From {}'.format(config.selector_dir))
    train_steps_per_epoch = len(train_dataset) // selector_params['batch_size']
    optimizer = AdamW(model.parameters(), lr=selector_params['lr'])
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=train_steps_per_epoch,
                                                num_training_steps=train_steps_per_epoch * selector_params['epoch'])
    logging.info('Starting Training')
    logging.info('Use the seed {}'.format(selector_params['seed']))
    train(train_loader, dev_loader, model, optimizer, scheduler, config.selector_model_dir)


if __name__ == '__main__':
    run()
