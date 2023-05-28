import torch
from transformers import BitsAndBytesConfig
from transformers import  AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel,prepare_model_for_kbit_training
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

def infer(args):
    model_id = "fnlp/moss-moon-003-sft"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config,trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto",trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)
    
    if args.lora_dir:
        print("Loading lora model from {}".format(args.lora_dir))
        model = PeftModel.from_pretrained(model,args.lora_dir,trust_remote_code=True)
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
