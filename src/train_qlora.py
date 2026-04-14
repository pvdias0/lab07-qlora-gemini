from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa fine-tuning com QLoRA.")
    parser.add_argument("--base-model", required=True, help="Modelo base no Hugging Face.")
    parser.add_argument("--train-file", required=True, help="Arquivo JSONL de treino.")
    parser.add_argument("--test-file", required=True, help="Arquivo JSONL de teste.")
    parser.add_argument("--output-dir", default="outputs/adapter", help="Diretorio de saida.")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Comprimento maximo.")
    parser.add_argument("--num-train-epochs", type=float, default=3.0, help="Epocas de treino.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="Batch de treino.")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1, help="Batch de avaliacao.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Acumulacao de gradiente.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--logging-steps", type=int, default=10, help="Intervalo de logs.")
    parser.add_argument("--save-steps", type=int, default=100, help="Intervalo de checkpoints.")
    return parser.parse_args()


def load_jsonl(path: str) -> Dataset:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"Nenhum registro encontrado em {path}.")
    return Dataset.from_list(records)


def ensure_text_field(dataset: Dataset) -> Dataset:
    if "text" in dataset.column_names:
        return dataset
    return dataset.map(
        lambda row: {
            "text": f"### Instrucao:\n{row['instruction']}\n\n### Resposta:\n{row['response']}"
        }
    )


def main() -> None:
    args = parse_args()

    train_dataset = ensure_text_field(load_jsonl(args.train_file))
    eval_dataset = ensure_text_field(load_jsonl(args.test_file))

    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError(
            "QLoRA exige CUDA. O ambiente atual esta sem suporte CUDA no PyTorch. "
            "Instale um build CUDA do torch antes de executar o treino."
        )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=False,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules="all-linear",
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=True,
        report_to="none",
        eval_strategy="epoch",
        save_strategy="epoch",
        dataset_text_field="text",
        max_length=args.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        args=training_args,
    )

    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adaptador salvo em: {args.output_dir}")


if __name__ == "__main__":
    main()
