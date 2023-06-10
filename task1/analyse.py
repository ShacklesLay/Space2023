import jsonlines
from collections import Counter

train_dir = '../dataset/jsonl/subtask3/task2_train_modified.jsonl'

n_fragments = []
n_type = []
n_entity = 0
n_entity_cont = 0


def check(list):
    if list[-1] - list[0] + 1 == len(list):
        return True
    else:
        return False

length = []
with open(train_dir, 'r') as f:
    for idx, item in enumerate(jsonlines.Reader(f)):
        n_fragments.append(len(item['reasons']))
        reasons = item['reasons']
        for frag in reasons:
            n_type.append(frag['type'])
            for entity in frag['fragments']:
                n_entity += 1
                if check(entity['idxes']):
                    n_entity_cont += 1
                    length.append(entity['idxes'][-1] - entity['idxes'][0])
                    if entity['idxes'][-1] - entity['idxes'][0] > 30:
                        print(idx, entity['idxes'])
                else:
                    print('Error!')


print(Counter(n_fragments))
print(Counter(n_type))
print('{}/{}={}'.format(n_entity_cont, n_entity, n_entity_cont/n_entity))
print(max(length))

