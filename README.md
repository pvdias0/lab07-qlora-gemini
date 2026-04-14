# Laboratorio 07 - QLoRA com Gemini

Projeto base para o laboratorio de especializacao de LLMs com LoRA e QLoRA.

Substituicao aplicada em relacao ao enunciado:
- A etapa de geracao do dataset sintetico usa a API do Gemini em vez da API da OpenAI.

## Estrutura

```text
lab07-qlora-gemini/
  data/
    processed/
  outputs/
  notebooks/
  src/
    generate_dataset.py
    train_qlora.py
```

## Requisitos

- Python 3.10+
- GPU CUDA com memoria suficiente para o modelo escolhido
- Chave da API Gemini em `GEMINI_API_KEY`

## Instalacao

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

No Windows, execute os scripts pelo wrapper PowerShell do projeto para forcar UTF-8:

```powershell
.\run_generate.ps1 --domain "atendimento medico" --count 50 --train-ratio 0.9 --output-dir data/processed
.\run_train.ps1 --base-model "meta-llama/Llama-2-7b-hf" --train-file data/processed/train.jsonl --test-file data/processed/test.jsonl --output-dir outputs/adapter
```

## Geracao do dataset sintetico

Exemplo:

```bash
python src/generate_dataset.py ^
  --domain "atendimento medico" ^
  --count 50 ^
  --train-ratio 0.9 ^
  --output-dir data/processed
```

Arquivos gerados:
- `data/processed/dataset_full.jsonl`
- `data/processed/train.jsonl`
- `data/processed/test.jsonl`

O `.jsonl` final deve permanecer versionado no repositorio para atender ao enunciado.

## Fine-tuning com QLoRA

Exemplo:

```bash
python src/train_qlora.py ^
  --base-model "meta-llama/Llama-2-7b-hf" ^
  --train-file data/processed/train.jsonl ^
  --test-file data/processed/test.jsonl ^
  --output-dir outputs/adapter
```

## Colab/Kaggle

Como sua maquina local nao tem NVIDIA, a rota mais segura e treinar em Colab ou Kaggle com GPU.

Arquivos adicionados para isso:
- `requirements-colab.txt`
- `notebooks/colab_train_qlora.ipynb`

Fluxo recomendado:
1. envie este projeto para um repositorio GitHub
2. abra o notebook no Colab ou replique as celulas no Kaggle
3. troque `!git clone <SEU_REPOSITORIO_GITHUB_AQUI> repo` pelo link real do seu repositorio
4. rode o treino com o modelo padrao `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

Observacoes:
- esse modelo e mais realista para GPU gratuita e continua atendendo a ideia do laboratorio
- se a GPU disponivel tiver memoria suficiente, voce pode testar um modelo maior
- o dataset ja esta pronto em `data/processed/train.jsonl` e `data/processed/test.jsonl`

## Parametros exigidos pelo laboratorio

- Quantizacao em 4 bits com `nf4`
- `compute_dtype=float16`
- `task_type=CAUSAL_LM`
- `r=64`
- `lora_alpha=16`
- `lora_dropout=0.1`
- otimizador `paged_adamw_32bit`
- scheduler `cosine`
- `warmup_ratio=0.03`

## Observacoes

- O script salva apenas o adaptador LoRA no fim do treino.
- Ajuste o modelo base se sua GPU nao suportar um modelo de 7B.
- O modelo citado no PDF pode exigir aceite de licenca no Hugging Face.
- Marque a versao de entrega no GitHub com a tag `v1.0`.
- O treino QLoRA exige GPU CUDA. Se o `torch` instalado estiver CPU-only, sera necessario reinstalar a build CUDA do PyTorch.

## Nota obrigatoria de IA

Inserir no `README.md` final do repositorio:

`Partes geradas/complementadas com IA codex, revisadas por Pedro Vitor Dias`
