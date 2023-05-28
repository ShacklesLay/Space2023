#!/usr/bin/env bash
set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/saved_model/task1_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/task1_finetune.json"
fi
INSTRUCTION="给定一段描述空间的文本，其中包含空间语义异常的文本片段。请分析文本并找出其中的空间语义异常片段。空间语义异常是指描述物体、场景或事件在空间方面出现与常识或语义规则不符的表述。在给定的文本中，定位并标识出所有的空间语义异常片段。 \
提示：异常文本片段的选取有以下六种情况：\n \
6个文本片段形成两个完整的“空间实体(S)-空间方位(P)-事件(E)”三元组，对应的role为S1 P1 E1 S2 P2 E2。\n \
5个文本片段形成一个完整的S-P-E三元组和一个不完整的三元组，不完整的三元组缺省了某个role。\n \
4个文本片段形成两个不完整的S-P-E三元组，各缺省一个role。\n \
3个文本片段形成一个完整的S-P-E三元组，对应的role为S1 P1 E1。\n \
2个文本片段形成一个不完整的S-P-E三元组，缺省了某个role。\n \
1个文本片段，role的取值是S1、P1或E1。\n \
模型提供的判断结果应包含以下信息：role、text和idxes。role的取值包括S1、P1、E1、S2、P2和E2，其中S表示空间实体，P表示空间方位，E表示空间事件。当文本片段个数不大于3时，role的取值包括S1、P1和E1。text中包含了与role相对应的文本，而idxes则是文本中每个字符所对应的索引。输出的格式应为一个包含多个字典的列表，每个字典代表一个空间语义异常片段，包含role、text和idxes字段。请根据文本分析出的空间语义异常片段，按照给定的格式提供输出。" 

python3 -u finetune_qlora.py \
--data "${DATA_DIR}" \
--output_dir "${OUTPUT_DIR}" \
--instruction "${INSTRUCTION}" \
--epochs 1 \