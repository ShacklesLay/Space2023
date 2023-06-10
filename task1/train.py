import torch
import logging
import numpy as np
from torch import nn
from tqdm import tqdm
import config
import copy


def train_epoch(train_loader, model, optimizer, scheduler, epoch):

    model.train()
    train_loss = 0.0
    for idx, batch_sample in enumerate(tqdm(train_loader)):
        torch.cuda.empty_cache()
        batch_data, batch_label = batch_sample
        batch_mask = batch_data.gt(0)

        # pred = model(batch_data, batch_mask)
        outputs = model(batch_data, batch_mask, labels=batch_label)
        loss = outputs['loss']
        train_loss += loss
        model.zero_grad()
        loss.backward()
        # is it necessary to use the clip_grad_norm ?
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
        optimizer.step()
        scheduler.step()
    train_loss = float(train_loss) / len(train_loader)
    return train_loss


def train(train_loader, dev_loader, model, optimizer, scheduler, model_dir):
    best_f1 = 0.0
    patience_counter = 0
    for epoch in range(1, config.epoch + 1):

        train_loss = train_epoch(train_loader, model, optimizer, scheduler, epoch)
        if config.subtask == 'subtask1':
            val_metrics = evaluate_task1(dev_loader, model)
        elif config.subtask == 'subtask2':
            val_metrics = evaluate_task2(dev_loader, model)
        elif config.subtask == 'subtask3':
            val_metrics = evaluate_task3(dev_loader, model)

        val_f1 = val_metrics['F1']
        precision = val_metrics['precision']
        recall = val_metrics['recall']
        logging.info('Epoch: {}, Train_loss: {:6f}, Precision: {:.6f}, Recall: {:.6f}, F1: {:.6f},'.format(
            epoch, train_loss, precision, recall, val_f1))

        if val_f1 > best_f1:
            model.save_pretrained(model_dir)
            logging.info('Save best model!')
            if val_f1 - best_f1 < config.patience:
                patience_counter += 1
            else:
                patience_counter = 0
            best_f1 = val_f1
        else:
            patience_counter += 1
        if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch:
            logging.info('Best f1: {}'.format(best_f1))
            break
    logging.info('Training Finished!')

# score_f1 don't consider the role, the result will be higher


def cal_f1(A_l, A_r, B):
    _A = set([i for i in range(A_l, A_r + 1)])
    _B = set(B)
    _intersection = _A & _B
    precision = len(_intersection) / len(_A)
    recall = len(_intersection) / len(_B)
    if precision == 0.0 or recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def score_f1(A, B, subtask):
    if A[0] > A[1]:
        A[0], A[1] = A[1], A[0]
    if A[2] > A[3]:
        A[2], A[3] = A[3], A[2]

    if subtask == 1:
        print(A[0], A[1], B[0])
        print(A[2], A[3], B[1])
        p1, r1, f1 = cal_f1(A[0], A[1], B[0])
        p2, r2, f2 = cal_f1(A[2], A[3], B[1])
        return (p1 + p2) / 2, (r1 + r2) / 2, (f1 + f2) / 2

    elif subtask == 2:
        for i in range(4, 12, 2):
            if A[i] > A[i+1]:
                A[i], A[i+1] = A[i+1], A[i]
        sum_p, sum_r, sum_f1, count = 0.0, 0.0, 0.0, 0
        for i in range(0, 12, 2):
            if B[i//2]:
                p, r, f1 = cal_f1(A[i], A[i+1], B[i//2])
                sum_p, sum_r, sum_f1, count = sum_p + p, sum_r + r, sum_f1 + f1, count + 1
        return sum_p/count, sum_r/count, sum_f1/count

    elif subtask == 3:
        if A[4] > A[5]:
            A[4], A[5] = A[5], A[4]
        print(A[0], A[1], B[0])
        print(A[2], A[3], B[1])
        print(A[4], A[5], B[2])
        sum_p, sum_r, sum_f1 = 0.0, 0.0, 0.0
        count = 0
        if B[0]:
            p, r, f1 = cal_f1(A[0], A[1], B[0])
            sum_p, sum_r, sum_f1, count = sum_p + p, sum_r + r, sum_f1 + f1, count + 1
        if B[1]:
            p, r, f1 = cal_f1(A[2], A[3], B[1])
            sum_p, sum_r, sum_f1, count = sum_p + p, sum_r + r, sum_f1 + f1, count + 1
        if B[2]:
            p, r, f1 = cal_f1(A[4], A[5], B[2])
            sum_p, sum_r, sum_f1, count = sum_p + p, sum_r + r, sum_f1 + f1, count + 1
        return sum_p/count, sum_r/count, sum_f1/count


def evaluate_task1(dev_loader, model):
    model.eval()
    P, R, F1 = 0.0, 0.0, 0.0
    with torch.no_grad():
        for idx, batch_sample in enumerate(tqdm(dev_loader)):
            batch_data, batch_label = batch_sample
            batch_mask = batch_data.gt(0)
            outputs = model(batch_data, batch_mask)
            # how to decode
            A_start = torch.argmax(outputs['A_start_logits'], dim=-1, keepdim=True)
            A_end = torch.argmax(outputs['A_end_logits'], dim=-1, keepdim=True)
            B_start = torch.argmax(outputs['B_start_logits'], dim=-1, keepdim=True)
            B_end = torch.argmax(outputs['B_end_logits'], dim=-1, keepdim=True)
            postion = torch.cat((A_start, A_end, B_start, B_end), dim=-1)
            sum_P, sum_R, sum_F1 = 0.0, 0.0, 0.0
            for B in range(batch_data.size(0)):
                maxp, maxr, maxf1 = 0.0, 0.0, 0.0
                blocks = len(batch_label[B]) // 2
                label = batch_label[B]
                for b in range(blocks):
                    p, r, f1 = score_f1(postion[B], label[2*b:2*(b+1)], subtask=1)
                    if f1 > maxf1:
                        maxp, maxr, maxf1 = p, r, f1
                sum_P, sum_R, sum_F1 = sum_P + maxp, sum_R + maxr, sum_F1 + maxf1
            sum_P = sum_P / batch_data.size(0)
            sum_R = sum_R / batch_data.size(0)
            sum_F1 = sum_F1 / batch_data.size(0)
            P, R, F1 = P + sum_P, R + sum_R, F1 + sum_F1
            '''
            for B in range(batch_data.size(0)):

                A_start = outputs['A_start_logits'][B]
                A_end = outputs['A_end_logits'][B]
                B_start = outputs['B_start_logits'][B]
                B_end = outputs['B_end_logits'][B]

                # max P(A,B) + P(C,D) ?
                # max P(A,B) * P(C,D) ?
                P_max = 0.0
                Best_postion = []
                len_sentence = batch_data.size(1)

                for A_l in range(len_sentence):
                    for A_r in range(A_l, min(len_sentence, A_l + 6)):
                        for B_l in range(A_r, len_sentence):
                            for B_r in range(B_l, min(len_sentence, B_l + 6)):
                                if A_start[A_l] * A_end[A_r] + B_start[B_l] * B_end[B_r] > P_max:
                                    P_max = A_start[A_l] * A_end[A_r] + B_start[B_l] * B_end[B_r]
                                    Best_postion = copy.deepcopy([A_l, A_r, B_l, B_r])

                p, r, f = score_f1(Best_postion, batch_label[B])
                sum_p, sum_r, sum_f1 = sum_p + p, sum_r + r, sum_f1 + f
            ave_p, ave_r, ave_f1 = sum_p / batch_data.size(0), sum_r / batch_data.size(0), sum_f1 / batch_data.size(0)
            P, R, F = P + ave_p, R + ave_r, F + ave_f1
        P, R, F = P / len(dev_loader), R / len(dev_loader), F / len(dev_loader)
        '''
    return {
        # fix the length of dev dataset
        'precision': P / len(dev_loader),
        'recall': R / len(dev_loader),
        'F1': F1 / len(dev_loader),
    }

def evaluate_task2(dev_loader, model):
    model.eval()
    P, R, F1 = 0.0, 0.0, 0.0
    with torch.no_grad():
        for idx, batch_sample in enumerate(tqdm(dev_loader)):

            batch_data, batch_label = batch_sample
            batch_mask = batch_data.gt(0)
            outputs = model(batch_data, batch_mask)

            # how to decode
            logits = []
            for i in range(12):
                logits.append(torch.argmax(outputs['logits'][:, :, i], dim=-1, keepdim=True))
            position = torch.cat(logits, dim=-1)
            sum_P, sum_R, sum_F1 = 0.0, 0.0, 0.0
            for B in range(batch_data.size(0)):
                maxp, maxr, maxf1 = 0.0, 0.0, 0.0
                label = batch_label[B]
                blocks = len(label) // 6
                for b in range(blocks):
                    p, r, f1 = score_f1(position[B], label[6*b: 6*(b+1)], subtask=2)
                    if f1 > maxf1:
                        maxp, maxr, maxf1 = p, r, f1
                sum_P, sum_R, sum_F1 = sum_P + maxp, sum_R + maxr, sum_F1 + maxf1

            sum_P = sum_P / batch_data.size(0)
            sum_R = sum_R / batch_data.size(0)
            sum_F1 = sum_F1 / batch_data.size(0)
            P, R, F1 = P + sum_P, R + sum_R, F1 + sum_F1

    return {
        'precision': P / len(dev_loader),
        'recall': R / len(dev_loader),
        'F1': F1 / len(dev_loader),
    }



def evaluate_task3(dev_loader, model):
    model.eval()
    P, R, F1 = 0.0, 0.0, 0.0
    with torch.no_grad():
        for idx, batch_sample in enumerate(tqdm(dev_loader)):

            batch_data, batch_label = batch_sample
            batch_mask = batch_data.gt(0)
            outputs = model(batch_data, batch_mask)

            # how to decode
            S_start = torch.argmax(outputs['S_start_logits'], dim=-1, keepdim=True)
            S_end = torch.argmax(outputs['S_end_logits'], dim=-1, keepdim=True)
            P_start = torch.argmax(outputs['P_start_logits'], dim=-1, keepdim=True)
            P_end = torch.argmax(outputs['P_end_logits'], dim=-1, keepdim=True)
            E_start = torch.argmax(outputs['E_start_logits'], dim=-1, keepdim=True)
            E_end = torch.argmax(outputs['E_end_logits'], dim=-1, keepdim=True)

            position = torch.cat((S_start, S_end, P_start, P_end, E_start, E_end), dim=-1)
            sum_P, sum_R, sum_F1 = 0.0, 0.0, 0.0
            for B in range(batch_data.size(0)):
                maxp, maxr, maxf1 = 0.0, 0.0, 0.0
                label = batch_label[B]
                blocks = len(label) // 3
                for b in range(blocks):
                    p, r, f1 = score_f1(position[B], label[3*b: 3*(b+1)], subtask=3)
                    if f1 > maxf1:
                        maxp, maxr, maxf1 = p, r, f1
                sum_P, sum_R, sum_F1 = sum_P + maxp, sum_R + maxr, sum_F1 + maxf1

            sum_P = sum_P / batch_data.size(0)
            sum_R = sum_R / batch_data.size(0)
            sum_F1 = sum_F1 / batch_data.size(0)
            P, R, F1 = P + sum_P, R + sum_R, F1 + sum_F1

    return {
        'precision': P / len(dev_loader),
        'recall': R / len(dev_loader),
        'F1': F1 / len(dev_loader),
    }


if __name__ == '__main__':
    A = [1, 3, 5, 8]
    B = [1, 4, 5, 6]

    print(score_f1(A,B))






