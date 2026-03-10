import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def train():
    # 1. Config
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    dataset_path = "training/dataset.jsonl"
    output_dir = "ml/models/sentience_core_lora"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        return

    # 2. Load Dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def format_instruction(sample):
        return f"<|user|>\n{sample['instruction']}\n{sample['input']}<|end|>\n<|assistant|>\n{sample['output']}<|end|>"

    # 3. BitsAndBytes Config (4-bit quantization)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 4. Load Model and Tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 5. LoRA Config
    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # 6. Set Training Arguments
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        fp16=True,
        max_grad_norm=0.3,
        num_train_epochs=3,
        weight_decay=0.001,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    # 7. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text", # We'll use formatting function instead
        formatting_func=format_instruction,
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    # 8. Train
    print("Starting fine-tuning...")
    trainer.train()

    # 9. Save Adapter
    trainer.model.save_pretrained(output_dir)
    print(f"Fine-tuning complete. Adapter saved to {output_dir}")

if __name__ == "__main__":
    train()
