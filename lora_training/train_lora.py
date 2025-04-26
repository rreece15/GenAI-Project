import argparse
import os

import torch
from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch


bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def parse_args():
    p = argparse.ArgumentParser("LoRA finetune with HF dataset names")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--adapter_name",         type=str, required=True)
    p.add_argument("--output_dir",           type=str, required=True)
    p.add_argument("--datasets",             nargs='+', required=True,
                   help="HF dataset identifiers; e.g. gsm8k or wikitext/wikitext-2-raw-v1")
    p.add_argument("--per_device_batch_size",    type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs",     type=int, default=3)
    p.add_argument("--learning_rate",        type=float, default=2e-4)
    return p.parse_args()


def load_and_format(name, tokenizer):
    # name may include a config: "dataset_name/config"
    if '/' in name:
        ds_name, ds_config = name.split('/', 1)
    else:
        ds_name, ds_config = name, None

    ds = load_dataset(ds_name, ds_config, split="train")
    cols = set(ds.column_names)

    def formatter(ex):
        if {"question", "answer"}.issubset(cols):
            prompt = f"Q: {ex['question'].strip()}\nA: {ex['answer'].strip()}"
        elif "text" in cols:
            prompt = ex["text"]
        else:
            raise ValueError(f"Dataset {name} has no question/answer or text fields")
        return tokenizer(prompt, truncation=True, max_length=512)

    ds = ds.map(formatter, batched=False, remove_columns=ds.column_names)
    return ds


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        quantization_config=bnb_cfg,
        trust_remote_code=True,      # needed for Llama‑4
        device_map="auto",           # auto shard across GPU/CPU
        offload_folder="offload",    # spill CPU tensors here
        offload_state_dict=True,     # keep full state dict off‑GPU
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = prepare_model_for_kbit_training(model)

    # attach LoRA
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # load all datasets
    print("Loading and formatting datasets:", args.datasets)
    all_ds = [load_and_format(ds, tokenizer) for ds in args.datasets]
    ds = concatenate_datasets(all_ds)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=50,
        save_total_limit=3,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
    )
    trainer.train()

    # save
    print("Saving adapter…")
    model.save_adapter(args.output_dir, adapter_name=args.adapter_name)
    print("Done.")


if __name__ == "__main__":
    main()
