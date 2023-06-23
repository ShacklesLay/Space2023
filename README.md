#  Space2023
# 文件介绍

`finetune.py`，`inference.py`参考[PhoebusSi/Alpaca-CoT](https://github.com/PhoebusSi/Alpaca-CoT)，分别用于lora微调以及加载lora模型进行推理。

`finetune_qlora.py`，`inference_qlora.py`使用[QLoRA](https://github.com/artidoro/qlora)进行微调和推理。

`process.ipynb`用于处理数据。

`scripts/ `内的脚本用于调用`finutune.py`文件进行微调，微调使用的参数以及instruction见相应脚本。

`task1`和`task2`文件夹里的内容为最终提交的任务一和任务二的相关代码。

请将任务一和任务二的数据文件放在`./data`中。

# 模型
MOSS模型来自[OpenLMLab/MOSS](https://github.com/OpenLMLab/MOSS)。

任务一的微调模型已上传至 https://huggingface.co/ShacklesLay/Deberta4task1

任务二的微调模型已上传至 https://huggingface.co/ShacklesLay/Deberta4task2
https://huggingface.co/ShacklesLay/CPT4task2
