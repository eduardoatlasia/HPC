#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Continued pretraining em PORTUGUÊS com Qwen3-1.7B + LoRA em 16 bits.

- Modelo: Qwen/Qwen3-1.7B
- Corpus: OSCAR (unshuffled_deduplicated_pt)
- Tarefa: causal language modeling (pré-treino continuado)
- LoRA: r=16 em projeções de atenção + MLP
- Saída: checkpoints em HPC/output/model/qwen3_1_7b_pt_lora
"""

import os
import math
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


# -------------------------------------------------------------------------
# CONFIGURAÇÕES GERAIS
# -------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-1.7B"           # << aqui trocamos o modelo
BLOCK_SIZE = 1024                        # tamanho do contexto em tokens
MAX_TRAIN_SAMPLES = None                 # ex: 10_000 para teste rápido
MAX_EVAL_SAMPLES = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

OUTPUT_MODEL_DIR = os.path.join(
    PROJECT_ROOT, "output", "model", "qwen3_1_7b_pt_lora"
)
OUTPUT_LOG_DIR = os.path.join(
    PROJECT_ROOT, "output", "logs", "qwen3_1_7b_pt_lora"
)

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_LOG_DIR, exist_ok=True)


# -------------------------------------------------------------------------
# TOKENIZER
# -------------------------------------------------------------------------

print(">> Carregando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# -------------------------------------------------------------------------
# MODELO BASE + LORA (16 bits, sem quantização)
# -------------------------------------------------------------------------

print(">> Carregando modelo base:", MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,   # 16 bits (bfloat16) se GPU suportar
    device_map="auto",
)

# Configuração LoRA (igual à do Qwen2, nomes de módulos tendem a ser os mesmos)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

print(">> Aplicando LoRA...")
model = get_peft_model(model, lora_config)
model.config.use_cache = False

print(">> Parâmetros treináveis:")
model.print_trainable_parameters()


# -------------------------------------------------------------------------
# DATASET: OSCAR (Português)
# -------------------------------------------------------------------------

print(">> Carregando dataset OSCAR (Português)...")
raw_datasets = load_dataset("oscar", "unshuffled_deduplicated_pt")

if "validation" not in raw_datasets:
    split = raw_datasets["train"].train_test_split(
        test_size=0.001, seed=42
    )
    raw_datasets = {
        "train": split["train"],
        "validation": split["test"],
    }

train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["validation"]


def filter_empty(example):
    text = example.get("text", "").strip()
    return len(text) > 0


print(">> Removendo linhas vazias...")
train_dataset = train_dataset.filter(filter_empty)
eval_dataset = eval_dataset.filter(filter_empty)


# -------------------------------------------------------------------------
# TOKENIZAÇÃO + AGRUPAMENTO EM BLOCO
# -------------------------------------------------------------------------

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        add_special_tokens=True,
        truncation=False,
    )


print(">> Tokenizando dataset...")
tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=train_dataset.column_names,
)
tokenized_eval = eval_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=eval_dataset.column_names,
)


def group_texts(examples):
    concatenated = []
    for input_ids in examples["input_ids"]:
        concatenated.extend(input_ids)

    total_length = (len(concatenated) // BLOCK_SIZE) * BLOCK_SIZE
    if total_length == 0:
        return {"input_ids": [], "attention_mask": []}

    concatenated = concatenated[:total_length]

    result = {
        "input_ids": [
            concatenated[i : i + BLOCK_SIZE]
            for i in range(0, total_length, BLOCK_SIZE)
        ]
    }
    result["attention_mask"] = [
        [1] * BLOCK_SIZE for _ in range(0, total_length, BLOCK_SIZE)
    ]
    return result


print(">> Agrupando tokens em blocos de tamanho", BLOCK_SIZE)
lm_train_dataset = tokenized_train.map(
    group_texts,
    batched=True,
    num_proc=4,
)
lm_eval_dataset = tokenized_eval.map(
    group_texts,
    batched=True,
    num_proc=4,
)

if MAX_TRAIN_SAMPLES:
    lm_train_dataset = lm_train_dataset.select(range(MAX_TRAIN_SAMPLES))
if MAX_EVAL_SAMPLES:
    lm_eval_dataset = lm_eval_dataset.select(range(MAX_EVAL_SAMPLES))


# -------------------------------------------------------------------------
# DATA COLLATOR (Causal LM)
# -------------------------------------------------------------------------

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# -------------------------------------------------------------------------
# TRAINING ARGUMENTS
# -------------------------------------------------------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_MODEL_DIR,
    overwrite_output_dir=True,
    num_train_epochs=1.0,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    weight_decay=0.01,
    logging_dir=OUTPUT_LOG_DIR,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    bf16=torch.cuda.is_available(),
    gradient_checkpointing=True,
    report_to="none",
)


# -------------------------------------------------------------------------
# TRAINER
# -------------------------------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_train_dataset,
    eval_dataset=lm_eval_dataset,
    data_collator=data_collator,
)


# -------------------------------------------------------------------------
# TREINO
# -------------------------------------------------------------------------

def main():
    print(">> Iniciando treino com Qwen3-1.7B...")
    train_result = trainer.train()
    trainer.save_model(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)

    metrics = train_result.metrics
    metrics["train_samples"] = len(lm_train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print(">> Avaliando...")
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(lm_eval_dataset)
    try:
        eval_loss = eval_metrics["eval_loss"]
        eval_metrics["perplexity"] = math.exp(eval_loss)
    except Exception:
        pass

    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print(">> Treino concluído.")
    if "perplexity" in eval_metrics:
        print(f">> Perplexidade de validação: {eval_metrics['perplexity']:.2f}")


if __name__ == "__main__":
    main()
