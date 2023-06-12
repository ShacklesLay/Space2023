import jsonlines

map_rr = ['空间实体',
    '参照实体',
    '事件',
    '事实性',
    '时间','时间','时间',
    '处所',
    '起点',
    '终点',
    '方向',
    '朝向',
    '部件处所',
    '部位',
    '形状',
    '路径',
'距离','距离']

def mapping(triple_18):
    triple = []
    for idx,element in enumerate(triple_18):
        if element is not None:
            new_element = {}
            role = map_rr[idx]
            new_element['role'] = role
            if type(element) == dict: 
                new_element['fragment'] = element
            elif type(element)==str:
                new_element['label']=element
            triple.append(new_element)
    return triple

data = []
file_path = "./experiments/deberata_cpt/prediction_test.jsonl"
with open(file_path,'r') as f:
    for item in jsonlines.Reader(f):
        a = {'qid':item['qid'],
             'context':item['context'],
            #  'corefs':item['corefs'],
            #  'non_corefs':item['non_corefs'],
             "results":[mapping(triple) for triple in item['outputs']],
             }
        data.append(a)
        
output_dir = "./processed_prediction_test.jsonl"
with jsonlines.open(output_dir, mode='w') as writer:
    for item in data:
        writer.write(item)