import os
import torch
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
import bitsandbytes as bnb
import argparse
from utils.device import DeviceMap  # 多卡微调

def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def get_accelarate_model(args):
    if args.full_finetune: assert args.bits in [16, 32]
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    model = AutoModelForCausalLM.from_pretrained(
        "fnlp/moss-moon-003-sft",
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        device_map=DeviceMap("Moss").get(), # 多卡微调，单卡设置为"auto"
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type # {'fp4', 'nf4'}
        ),
        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        trust_remote_code=args.trust_remote_code,
        )
    if compute_dtype == torch.float16 and args.bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)
            
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    modules = find_all_linear_names(args, model)
    
    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    
    if not args.full_finetune:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    print(f'adding LoRA modules...')
    model = get_peft_model(model, config)
    
    if args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    for name, module in model.named_modules():
        # if isinstance(module, LoraLayer):
        #     if args.bf16:
        #         module = module.to(torch.bfloat16)
        if 'ln' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model

def train(args):

    print("Instruction used for finetuning: ", args.instruction)    
    
#     bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )
#     model = AutoModelForCausalLM.from_pretrained(
#         "fnlp/moss-moon-003-sft",
#         trust_remote_code=True,
#         quantization_config=bnb_config,
#         device_map={"":0}
#     )
#     model.gradient_checkpointing_enable()
#     model = prepare_model_for_kbit_training(model)

        
    # config = LoraConfig(
    #     r=args.lora_r,
    #     lora_alpha=args.lora_alpha,
    #     target_modules=["q_proj", "v_proj"],
    #     lora_dropout=args.lora_dropout,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # # )

    # model = get_peft_model(model, config)
    model = get_accelarate_model(args)

    tokenizer = AutoTokenizer.from_pretrained(
        "fnlp/moss-moon-003-sft", 
        add_eos_token=True, 
        trust_remote_code=True
    )
    tokenizer.eos_token_id = 106068 # The eos_token_id of base model is 106028. We need map the eos token to <eom> (its token id is 106068)
    tokenizer.pad_token_id = 0
    
    data = load_dataset(
        "json",
        data_files=args.data
    )

    train_val = data["train"].train_test_split(test_size=args.val_set_size,
                                            shuffle=True,
                                            seed=42)
    train_data = train_val["train"]
    val_data = train_val["test"]

    def generate_prompt(data_point):
        instruction = args.instruction
        input = data_point["input"]
        response = data_point["output"]

        return f"<|Human|>:\n{instruction}\n{input}\n<eoh>\n<|MOSS|>:{response}<eom>\n"


    def tokenize(prompt):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len + 1,
            padding="max_length",
        )
        return {
            "input_ids": result["input_ids"][:-1],
            "attention_mask": result["attention_mask"][:-1],
        }

    train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x)))
    val_data = val_data.shuffle().map(lambda x: tokenize(generate_prompt(x)))


    # saving_step = 4
    warmup_steps = 2

    print("***** Running training *****")
    print(f"  Num Epochs = {args.epochs}", )
    print(f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    # print(f"  Saving steps = {saving_step}")
    print(f"  Warmup steps = {warmup_steps}")
    model.print_trainable_parameters()
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.per_gpu_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=10,
            output_dir=args.output_dir,
            optim="paged_adamw_8bit"
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer,
                                                                mlm=False),
    )
    model.config.use_cache = False

    # old_state_dict = model.state_dict
    # model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(
    #     self, old_state_dict())).__get__(model, type(model))

    trainer.train()
    model.save_pretrained(args.output_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data', type=str, default="./data/trans_chinese_alpaca_data.json",help='the data used for instructing tuning')
    parser.add_argument('--per_gpu_train_batch_size', default=8, type=int, help='Batch size per GPU/CPU for training.')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--cutoff_len', default=256, type=int)
    
    #PEFT arguments
    parser.add_argument('--lora_r', default=8, type=int)
    parser.add_argument('--lora_alpha', default=16, type=int)
    parser.add_argument('--lora_dropout', default=0.05, type=float)
    parser.add_argument('--val_set_size', default=0.1, type=float)
    
    parser.add_argument('--output_dir', default='./saved_model', type=str)
    parser.add_argument('--instruction', default='', type=str)
    
    #Quantization arguments
    parser.add_argument('--bits', default=4, type=int)
    parser.add_argument('--full_finetune', default=False, type=bool)
    parser.add_argument('--fp16', default=False, type=bool,help="Whether to use fp16 for training, fp16 is priority")
    parser.add_argument('--bf16', default=True, type=bool,help="Whether to use bfloat16 for training")
    parser.add_argument('--double_quant', default=True, type=bool,help="Whether to use double quantization for training")
    parser.add_argument('--quant_type', default="nf4", type=str,choices=['fp4', 'nf4'],help="Whether to use double quantization for training")
    parser.add_argument('--trust_remote_code', default=True, type=bool)
    parser.add_argument('--gradient_checkpointing', default=True, type=bool)

    args, _ = parser.parse_known_args()
    print(args)

    train(args)
