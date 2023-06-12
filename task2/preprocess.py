import jsonlines

map_r = {
    '空间实体':0,
    '参照实体':1,
    '事件':2,
    '事实性':3,
    # '时间':4,5,6
    '处所':7,
    '起点':8,
    '终点':9,
    '方向':10,
    '朝向':11,
    '部件处所':12,
    '部位':13,
    '形状':14,
    '路径':14,
    # '距离':16,17
}
def mapping(triple):
    triple_18 = [None for _ in range(18)]
    for element in triple:
        role = element['role']
        if role == '时间':
            if element.get('fragment') is not None:
                triple_18[4] = element['fragment']
            if element.get('label') is not None:
                triple_18[6] = element['label']
        elif role=="距离":
            if element.get('fragment') is not None:
                triple_18[16] = element['fragment']
            if element.get('label') is not None:
                triple_18[17] = element['label']
        elif role=='事实性':
            triple_18[3]=element['label']
        else:
            triple_18[map_r[role]] = element['fragment']
    return triple_18

data = []
data_dir = "../data/"
file_name = "task2_train.jsonl"
file_path = data_dir + file_name
with open(file_path,'r') as f:
    for item in jsonlines.Reader(f):
        a = {'qid':item['qid'],
             'context':item['context'],
             'corefs':item['corefs'],
             'non_corefs':item['non_corefs'],
             'outputs':[mapping(triple) for triple in item['results']],
             }
        data.append(a)

output_dir = "../data/"
with jsonlines.open(output_dir+'processed_'+file_name, mode='w') as writer:
    for item in data:
        writer.write(item)
        
file_name = "task2_dev.jsonl"
file_path = data_dir + file_name
with open(file_path,'r') as f:
    for item in jsonlines.Reader(f):
        a = {'qid':item['qid'],
             'context':item['context'],
             'corefs':item['corefs'],
             'non_corefs':item['non_corefs'],
             'outputs':[mapping(triple) for triple in item['results']],
             }
        data.append(a)

output_dir = "../data/"
with jsonlines.open(output_dir+'processed_'+file_name, mode='w') as writer:
    for item in data:
        writer.write(item)