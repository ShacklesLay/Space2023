import os 
import torch
from MOSS.models.modeling_moss import MossForCausalLM
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from peft import PeftModel
import argparse

instruction = "文本中存在空间语义异常，为了描述空间语义异常，最多可以选取6个文本片段，最少可以选取1个。每个文本片段包含3个字段，分别是角色（role）、文本内容（text）和字序数组 （idxes）。role的取值包括S1,P1,E1,S2,P2,E2，当文本片段个数不大于3时，role的取值包括S1,P1,E1。异常文本片段的选取有以下六种情况：\n \
1.6 个文本片段。这些片段形成两个完整的“空间实体(S)-空间方位(P)-事件(E)”三元组，其对应的role分别为S1,P1,E1,S2,P2,E2。如“池水照见了她的面容和身影；她笑，池水里的影子也向着她笑；她假装生气，池水外的影子也向着她生气”，需要池水里的影子和池水外的影子共同描述空间语义异常。异常文本片段是S1 影子, P1 池水里, E1 笑, S2 影子, P2 池水外, E2 生气。 \
2.5 个文本片段。这些片段形成一个完整的S-P-E三元组和一个不完整的三元组，不完整的三元组缺省了某个role，该role在文本中没有出现。如“双腿上下交叉，右腿在上、左腿在旁”，没有出现表达左腿在旁的方式、目的等的核心谓词成分。异常文本片段是S1 双腿, P1 上下, E1 交叉, S2 左腿, P2 在旁。 \
3.4 个文本片段。这些片段形成两个不完整的S-P-E三元组，各缺省一个role，两个role在文本中没有出现。如“他钻了进去，于是我也钻了出来，与他大打一场”，该句没有出现表达P信息，所以不标注P1和P2。异常文本片段是S1 他, E1 钻进去, S2 我, E2 钻出来。 \
4.3 个文本片段。这些片段形成一个完整的S-P-E三元组，其对应的role分别为S1 P1 E1。如“一只手臂枕在脑袋上面”，不需要其他空间信息便可以判断出有空间语义异常，因为手臂只能枕在脑袋下面。异常文本片段是S1 手臂, P1 在脑袋上面, E1 枕。 \
5.2 个文本片段。这些片段形成一个不完整的S-P-E三元组，缺省了某个role。具体有以下两种情况： \
- 该role没有在文中出现。如“他上衣上面的后裙是多么美丽！”，没有出现表达后裙位于上衣上面的方式、目的等的核心谓词成分，所以不标注E1。异常文本片段是S1 后裙, P1 上衣后面； \
- 不需要该role便可以描述异常。如“他可以从缝隙钻进屋子外”，该异常发生在钻进屋子外，与空间实体他无关，所以不标注S1。异常文本片段是P1 屋子外, E1 钻进。 \
6.1 个文本片段，role的取值是S1、P1或E1，文本的某个空间实体、某个空间方位或某个事件存在异常。如“仰卧在地面边”，处所在地面边存在异常。异常文本片段是P1 在地面边。 \
仿照上述规则和案例，找出下列文本中含有空间异常的文本片段，输出格式为{ role: 角色, text: 文本内容, idxes: 文本内容的索引数组 }："
input = "德国人又开炮了，炮弹在这小小的方场上炸开了，黑色的泥土直翻起来，柱子似的。碎片把那些剩下来的树木的枝条都削去了。那个苏联人孤零零地躺在那毫无遮掩的方场上，一只手臂枕在脑袋上面，周围是炸弯了的铁器和炸焦了的树木。"
query = f"<|Human|>:\n{instruction}\n{input}\n<eoh>\n<|MOSS|>:"

# os.environ['CUDA_VISIBLE_DEVICES'] = "5,6"
model_path = "/remote-home/cktan/.cache/huggingface/hub/models--fnlp--moss-moon-003-sft/snapshots/7119d446173035561f40977fb9cb999995bb7517"
if not os.path.exists(model_path):
    model_path = snapshot_download(model_path)

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
    parser.add_argument('--lora_dir', type=str, default='./saved_model/task1_2023-05-10-1302.14', \
                                        help='the directory of lora model')
    
    args, _ = parser.parse_known_args()
    infer(args)