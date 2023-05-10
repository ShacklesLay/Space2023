import os
import torch
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
import argparse

from utils.device import DeviceMap

def train(args):

    world_size = torch.cuda.device_count()
    print(world_size)
    
    print("Instruction used for finetuning: ", args.instruction)

    tokenizer = AutoTokenizer.from_pretrained(
        "fnlp/moss-moon-003-sft", 
        add_eos_token=True, 
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        "fnlp/moss-moon-003-sft",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=DeviceMap("Moss").get()
    )


    if (world_size > 1):
        model.is_parallelizable = True
        model.model_parallel = True
        
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

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

        return f"<|Human|>:\n{instruction}\n{input}\n<eoh>\n<|MOSS|>:{response}"


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

    total_batch_size = args.per_gpu_train_batch_size * args.gradient_accumulation_steps * world_size
    total_optim_steps = train_data.num_rows // total_batch_size
    # saving_step = int(total_optim_steps/10)
    # warmup_steps = int(total_optim_steps/10)
    saving_step = 4
    warmup_steps = 2

    print("***** Running training *****")
    print(f"  Num Epochs = {args.epochs}", )
    print(f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Total optimization steps = {total_optim_steps}")
    print(f"  Saving steps = {saving_step}")
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
            logging_steps=1,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=saving_step,
            save_steps=saving_step,
            output_dir=args.output_dir,
            save_total_limit=3,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if world_size > 1 else None,
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
    parser.add_argument('--per_gpu_train_batch_size', default=4, type=int, help='Batch size per GPU/CPU for training.')
    parser.add_argument('--gradient_accumulation_steps', default=32, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--cutoff_len', default=256, type=int)
    #PEFT arguments
    parser.add_argument('--lora_r', default=8, type=int)
    parser.add_argument('--lora_alpha', default=16, type=int)
    parser.add_argument('--lora_dropout', default=0.05, type=float)
    parser.add_argument('--val_set_size', default=0.1, type=float)
    
    parser.add_argument('--output_dir', default='./saved_model', type=str)
    parser.add_argument('--instruction', default='', type=str)

    args, _ = parser.parse_known_args()
    print(args)

    train(args)
