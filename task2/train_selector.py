from tqdm import tqdm
import torch
import logging

from config import selector_params as params
import torch.nn as nn


def train_epoch(train_loader, model, optimizer, scheduler):
    model.train()
    train_loss = 0.0
    for _, batch_sample in enumerate(tqdm(train_loader)):
        batch_data, batch_label = batch_sample
        batch_mask = batch_data.gt(0)
        outputs = model(batch_data, batch_mask, labels=batch_label)
        loss = outputs['loss']
        train_loss += loss.item()
        model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=params['clip_grad'])
        optimizer.step()
        scheduler.step()
    train_loss = float(train_loss) / len(train_loader)
    return train_loss


def evaluate(dev_loader, model):
    model.eval()
    with torch.no_grad():
        P, R, F1 = [], [], []
        for _, batch_sample in enumerate(dev_loader):
            batch_data, batch_label = batch_sample
            batch_mask = batch_data.gt(0)
            outputs = model(batch_data, batch_mask)
            batch_pred = torch.argmax(outputs['logits'], dim=-1)
            len_batch = batch_label.size(0)
            for B in range(len_batch):

                len_mask = batch_mask[B].size(0)
                pred = batch_pred[B].cpu().numpy().tolist()[0:len_mask]
                label = batch_label[B].cpu().numpy().tolist()[0:len_mask]

                pred_idx = set([idx for idx, val in enumerate(pred) if val == 1])
                gold_idx = set([idx for idx, val in enumerate(label) if val == 1])
                intersection = pred_idx & gold_idx
                precision = len(intersection) / len(pred_idx) if len(pred_idx) > 0 else 0
                recall = len(intersection) / len(gold_idx)
                if precision == 0 or recall == 0 or intersection == 0:
                    f1 = 0
                else:
                    f1 = 2 * precision * recall / (precision + recall)
                P.append(precision)
                R.append(recall)
                F1.append(f1)

    return {
        'avg_precision': sum(P)/len(P),
        'avg_recall': sum(R)/len(R),
        'f1': sum(F1)/len(F1)
    }


def train(train_loader, dev_loader, model, optimizer, scheduler, model_dir):
    # metric f1:
    # p: |A and B| / |A|
    # r: |A and B| / |B|
    # f1 = 2 * p * r / (p + r)
    best_recall = 0.0
    patience_counter = 0
    for epoch in range(1, params['epoch'] + 1):
        train_loss = train_epoch(train_loader, model, optimizer, scheduler)
        metrics = evaluate(dev_loader, model)
        recall = metrics['avg_recall']
        precision = metrics['avg_precision']
        f1 = metrics['f1']
        logging.info('Epoch: {}, Train_loss: {}, Recall: {}, Precision: {}, F1: {}'.format(epoch, train_loss, recall, precision, f1))
        if recall > best_recall and epoch >= 5:
            model.save_pretrained(model_dir)
            logging.info('Model Saved!')
            logging.info('find the better recall score!')
            if recall - best_recall < params['patience']:
                patience_counter += 1
            else:
                patience_counter = 0
            best_recall = recall
        else:
            patience_counter += 1
        if (patience_counter >= params['patience_num'] and epoch > params['min_epoch_num']) or epoch == params['epoch']:
            logging.info('Best recall: {}'.format(best_recall))
            break
    logging.info('Training Finished!')
