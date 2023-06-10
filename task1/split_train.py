import os
import json
import jsonlines
from config import original_dir, train_dir, dev_dir, test_dir, subtask_dir


def sample():
    processed_data = []

    with open(original_dir, 'r') as fr:
        for idx, item in enumerate(jsonlines.Reader(fr)):
            processed_reasons = []
            qid, context, reasons = item['qid'], item['context'], item['reasons']
            for fragments in reasons:
                if fragments['type'] == 'B':
                    processed_reasons.append({'fragments': fragments['fragments'],
                                              'type': fragments['type']})
            if len(processed_reasons) == 0:
                continue
            else:
                processed_data.append({'qid': qid,
                                       'context': context,
                                       'reasons': processed_reasons})
            # if len(processed_reasons) > 1:
            #     print(processed_reasons)

    with jsonlines.open(subtask_dir, 'w') as f:
        f.write_all(processed_data)


def split():

    train, dev, test = [], [], []

    with open(subtask_dir, 'r') as f:
        total = 1258
        # total = len(list(jsonlines.Reader(f)))
        # print(total)
        # how to get the length of train/transfer it into list
        count = 0

        for idx, item in enumerate(jsonlines.Reader(f)):
            # print('into it')
            context, reasons = item['context'], item['reasons']
            count += 1

            if count < int(0.8 * total):
                train.append({'context': context, 'reasons': reasons})
            elif count < int(0.9 * total):
                dev.append({'context': context, 'reasons': reasons})
            else:
                test.append({'context': context})

    # print(len(train), len(dev), len(test))
    with jsonlines.open(train_dir, 'w') as train_f:
        train_f.write_all(train)
    with jsonlines.open(dev_dir, 'w') as dev_f:
        dev_f.write_all(dev)
    with jsonlines.open(test_dir, 'w') as test_f:
        test_f.write_all(test)


if __name__ == '__main__':
    sample()
    # split()

