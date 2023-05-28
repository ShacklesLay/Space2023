import os 
import torch
from MOSS.models.modeling_moss import MossForCausalLM
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from peft import PeftModel
import argparse

instruction = "给定一段描述空间的文本，其中包含空间语义异常的文本片段。请分析文本并找出其中的空间语义异常片段。空间语义异常是指描述物体、场景或事件在空间方面出现与常识或语义规则不符的表述。在给定的文本中，定位并标识出所有的空间语义异常片段。 \
提示：异常文本片段的选取有以下六种情况：\n \
6个文本片段形成两个完整的“空间实体(S)-空间方位(P)-事件(E)”三元组，对应的role为S1 P1 E1 S2 P2 E2。\n \
5个文本片段形成一个完整的S-P-E三元组和一个不完整的三元组，不完整的三元组缺省了某个role。\n \
4个文本片段形成两个不完整的S-P-E三元组，各缺省一个role。\n \
3个文本片段形成一个完整的S-P-E三元组，对应的role为S1 P1 E1。\n \
2个文本片段形成一个不完整的S-P-E三元组，缺省了某个role。\n \
1个文本片段，role的取值是S1、P1或E1。\n \
模型提供的判断结果应包含以下信息：role、text和idxes。role的取值包括S1、P1、E1、S2、P2和E2，其中S表示空间实体，P表示空间方位，E表示空间事件。当文本片段个数不大于3时，role的取值包括S1、P1和E1。text中包含了与role相对应的文本，而idxes则是文本中每个字符所对应的索引。输出的格式应为一个包含多个字典的列表，每个字典代表一个空间语义异常片段，包含role、text和idxes字段。请根据文本分析出的空间语义异常片段，按照给定的格式提供输出。" 
input = "德国人又开炮了，炮弹在这小小的方场上炸开了，黑色的泥土直翻起来，柱子似的。碎片把那些剩下来的树木的枝条都削去了。那个苏联人孤零零地躺在那毫无遮掩的方场上，一只手臂枕在脑袋上面，周围是炸弯了的铁器和炸焦了的树木。"
query = f"<|Human|>:\n{instruction}\ninput:{input}\n<eoh>\n<|MOSS|>:"
# query = "给定一段描述空间的文本，其中包含空间语义异常的文本片段。请分析文本并找出其中的空间语义异常片段。空间语义异常是指描述物体、场景或事件在空间方面出现与常识或语义规则不符的表述。在给定的文本中，定位并标识出所有的空间语义异常片段。 \
# 提示：异常文本片段的选取有以下六种情况：\n \
# 6个文本片段形成两个完整的“空间实体(S)-空间方位(P)-事件(E)”三元组，对应的role为S1 P1 E1 S2 P2 E2。\n \
# 5个文本片段形成一个完整的S-P-E三元组和一个不完整的三元组，不完整的三元组缺省了某个role。\n \
# 4个文本片段形成两个不完整的S-P-E三元组，各缺省一个role。\n \
# 3个文本片段形成一个完整的S-P-E三元组，对应的role为S1 P1 E1。\n \
# 2个文本片段形成一个不完整的S-P-E三元组，缺省了某个role。\n \
# 1个文本片段，role的取值是S1、P1或E1。\n \
# 模型提供的判断结果应包含以下信息：role、text和idxes。role的取值包括S1、P1、E1、S2、P2和E2，其中S表示空间实体，P表示空间方位，E表示空间事件。当文本片段个数不大于3时，role的取值包括S1、P1和E1。text中包含了与role相对应的文本，而idxes则是文本中每个字符所对应的索引。输出的格式应为一个包含多个字典的列表，每个字典代表一个空间语义异常片段，包含role、text和idxes字段。请根据文本分析出的空间语义异常片段，按照给定的格式提供输出。 \
# 以下是按照上述规则从输入文本中找出包含空间语义异常的文本片段的例子：\n \
# <input>德国人又开炮了，炮弹在这小小的方场上炸开了，黑色的泥土直翻起来，柱子似的。碎片把那些剩下来的树木的枝条都削去了。那个苏联人孤零零地躺在那毫无遮掩的方场上，一只手臂枕在脑袋上面，周围是炸弯了的铁器和炸焦了的树木。\n \
# <output>[{'role': 'S1', 'text': '手臂', 'idxes': [79, 80]}, {'role': 'P1', 'text': '在脑袋上面', 'idxes': [82, 83, 84, 85, 86]}, {'role': 'E1', 'text': '枕', 'idxes': [81]}]\n \
# <intput>她穿过方场，到了那战死的苏联士兵身边，她用力把那尸身翻过来。看见他的面孔了，很年轻，很苍白。她轻轻理好了他的头发，又费了很大的劲把他那一双早已僵硬了的手臂弯过来，交叉地覆在他的胸前。然后，她在他旁边坐了上来。\n \
# <output>[{'role': 'S1', 'text': '她', 'idxes': [94]}, {'role': 'P1', 'text': '在他旁边', 'idxes': [95, 96, 97, 98]}, {'role': 'E1', 'text': '坐了上来', 'idxes': [99, 100, 101, 102]}]\n \
# <input>她终于站了起来，离开了那死者。走了不多几步，她马上找到她要的东西了：一个大的炮弹坑。这是几天之前炸进来的，现在，那坑里已经积了些水。\n \
# <output>[{'role': 'S1', 'text': '炮弹坑', 'idxes': [38, 39, 40]}, {'role': 'E1', 'text': '炸进来', 'idxes': [48, 49, 50]}]\n \
# 按照上述规则，仿照例子，从下列文本中找出包含空间语义异常的文本片段：\n \
# <intput>桑桑带着柳柳来到城墙下时，已近黄昏。桑桑仰望着这堵高得似乎要碰到了天的城墙，心里很激动。他要带着柳柳沿着台阶登到城墙顶后，但柳柳走不动了。他让柳柳坐在了台阶上，然后脱掉了柳柳脚上的鞋。他看到柳柳的脚板底打了两个豆粒大的血泡。他轻轻地揉了揉她的脚，给她穿上鞋，蹲下来，对她说:“哥哥背你上去。”\n \
# <output>"

# os.environ['CUDA_VISIBLE_DEVICES'] = "5,6"
# model_path = "/remote-home/cktan/.cache/huggingface/hub/models--fnlp--moss-moon-003-sft/snapshots/7119d446173035561f40977fb9cb999995bb7517"
model_path = "fnlp/moss-moon-003-sft"
if not os.path.exists(model_path):
    model_path = snapshot_download(model_path,resume_download=True)

def infer(args):
    config = AutoConfig.from_pretrained("fnlp/moss-moon-003-sft", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("fnlp/moss-moon-003-sft", trust_remote_code=True)
    with init_empty_weights():
        model = MossForCausalLM._from_config(config, torch_dtype=torch.float16)
    model.tie_weights()
    model = load_checkpoint_and_dispatch(model, model_path, device_map="auto", no_split_module_classes=["MossBlock"], dtype=torch.float16)
    if args.lora_dir:
        model = PeftModel.from_pretrained(model,args.lora_dir)
        print("load lora model from {}".format(args.lora_dir))

    inputs = tokenizer(query, return_tensors="pt")
    for k in inputs:
        inputs[k] = inputs[k].cuda()
    outputs = model.generate(**inputs, do_sample=True, temperature=0.7, top_p=0.8, repetition_penalty=1.02, max_new_tokens=256)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print('Response:')
    print(response)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--lora_dir', type=str, default='./saved_model/task1_2023-05-25-1403.34/', \
                                        help='the directory of lora model')
    
    args, _ = parser.parse_known_args()
    infer(args)