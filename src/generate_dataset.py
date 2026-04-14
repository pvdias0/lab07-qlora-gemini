from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai


SYSTEM_PROMPT = """
Voce e um gerador de datasets de instrucao para fine-tuning supervisionado.
Gere exemplos no formato JSON.
Cada item precisa ter:
- instruction: pergunta ou instrucao do usuario
- response: resposta ideal, objetiva e correta

Regras:
- use portugues do Brasil
- varie a dificuldade
- evite repeticoes
- nao inclua texto fora do JSON
- retorne apenas uma lista JSON valida
""".strip()


@dataclass
class Example:
    instruction: str
    response: str

    def to_record(self) -> dict[str, str]:
        return {
            "instruction": self.instruction.strip(),
            "response": self.response.strip(),
            "text": f"### Instrucao:\n{self.instruction.strip()}\n\n### Resposta:\n{self.response.strip()}",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera dataset sintetico com Gemini.")
    parser.add_argument("--domain", required=True, help="Dominio do dataset sintetico.")
    parser.add_argument("--count", type=int, default=50, help="Quantidade total de exemplos.")
    parser.add_argument("--batch-size", type=int, default=10, help="Quantidade de exemplos por chamada.")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Fracao destinada ao treino.")
    parser.add_argument("--seed", type=int, default=42, help="Seed para embaralhamento.")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Modelo Gemini a ser usado.")
    parser.add_argument("--output-dir", default="data/processed", help="Diretorio de saida.")
    parser.add_argument("--max-attempts", type=int, default=20, help="Limite de tentativas para fechar o dataset.")
    return parser.parse_args()


def require_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Defina GEMINI_API_KEY no ambiente antes de executar o script.")
    return api_key


def build_user_prompt(domain: str, amount: int, offset: int) -> str:
    return f"""
Gere {amount} exemplos para um dataset de instrucoes no dominio "{domain}".
Os exemplos devem cobrir cenarios praticos, conceituais, explicativos e orientados a tarefa.
Evite duplicatas com exemplos anteriores; o lote atual comeca no indice {offset}.

Formato de saida:
[
  {{
    "instruction": "...",
    "response": "..."
  }}
]
""".strip()


def extract_json_array(text: str) -> list[dict[str, Any]]:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("A resposta do modelo nao contem uma lista JSON valida.")
    payload = text[start : end + 1]
    data = json.loads(payload)
    if not isinstance(data, list):
        raise ValueError("A resposta do modelo nao retornou uma lista.")
    return data


def request_batch(client: genai.Client, model: str, domain: str, amount: int, offset: int) -> list[Example]:
    response = client.models.generate_content(
        model=model,
        contents=build_user_prompt(domain, amount, offset),
        config={
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0.8,
            "response_mime_type": "application/json",
        },
    )
    items = extract_json_array(response.text)
    batch: list[Example] = []
    for item in items:
        instruction = str(item.get("instruction", "")).strip()
        answer = str(item.get("response", "")).strip()
        if instruction and answer:
            batch.append(Example(instruction=instruction, response=answer))
    if not batch:
        raise ValueError("Nenhum exemplo valido foi retornado pelo Gemini.")
    return batch


def deduplicate(examples: list[Example]) -> list[Example]:
    seen: set[tuple[str, str]] = set()
    unique: list[Example] = []
    for item in examples:
        key = (item.instruction.casefold(), item.response.casefold())
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.count < 50:
        raise ValueError("O enunciado exige pelo menos 50 pares de prompt/response.")
    if not 0 < args.train_ratio < 1:
        raise ValueError("train-ratio deve estar entre 0 e 1.")

    client = genai.Client(api_key=require_api_key())
    collected: list[Example] = []
    attempts = 0

    while len(collected) < args.count and attempts < args.max_attempts:
        attempts += 1
        remaining = args.count - len(collected)
        batch_size = min(args.batch_size, remaining)
        try:
            batch = request_batch(
                client=client,
                model=args.model,
                domain=args.domain,
                amount=batch_size,
                offset=len(collected),
            )
        except Exception as exc:
            print(f"Tentativa {attempts} falhou: {exc}")
            time.sleep(2)
            continue
        collected.extend(batch)
        collected = deduplicate(collected)
        print(f"Exemplos acumulados: {len(collected)}/{args.count}")
        time.sleep(1)

    if len(collected) < args.count:
        raise RuntimeError(
            f"Nao foi possivel atingir {args.count} exemplos validos apos {args.max_attempts} tentativas."
        )

    records = [item.to_record() for item in collected[: args.count]]

    rng = random.Random(args.seed)
    rng.shuffle(records)

    train_size = int(len(records) * args.train_ratio)
    train_records = records[:train_size]
    test_records = records[train_size:]

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "dataset_full.jsonl", records)
    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "test.jsonl", test_records)

    print(f"Dataset completo salvo em: {output_dir / 'dataset_full.jsonl'}")
    print(f"Treino: {len(train_records)} exemplos")
    print(f"Teste: {len(test_records)} exemplos")


if __name__ == "__main__":
    main()
