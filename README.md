<h1 align="center">🦙 Llama-3 8B Unsloth Fine-Tuning</h1>

<p align="center">
  <b>Fine-tuning Meta's Llama-3 8B model with Unsloth for TRObject code generation</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-Llama--3%208B-blue?logo=meta" alt="Model">
  <img src="https://img.shields.io/badge/Framework-Unsloth-orange" alt="Unsloth">
  <img src="https://img.shields.io/badge/Quantization-4bit%20QLoRA-green" alt="4bit QLoRA">
  <img src="https://img.shields.io/badge/Export-GGUF%20q4__k__m-purple" alt="GGUF">
  <img src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License">
</p>

---

## 📖 Overview

This project demonstrates how to fine-tune **Meta's Llama-3 8B** large language model on a custom Turkish dataset using the [Unsloth](https://github.com/unslothai/unsloth) framework. The fine-tuned model specialises in generating **TRObject** framework code from natural-language instructions written in Turkish.

The repository contains:

| File | Description |
|---|---|
| `Llama-3 8b Unsloth Fine-tuning_Mehmet ÖDEN.ipynb` | Full fine-tuning pipeline (load → LoRA → train → export) |
| `TRObjectLlama_Mehmet ÖDEN.ipynb` | Deployment examples (FastAPI REST API & Tkinter desktop app) |
| `Llama-3 8b Unsloth Fine-tuning_Mehmet ÖDEN.pdf` | PDF version of the fine-tuning notebook |

---

## ✨ Features

- ⚡ **2× faster training** with Unsloth's optimised CUDA kernels
- 🧠 **4-bit QLoRA** quantisation – fits in a single consumer GPU
- 🔧 **LoRA adapters** – trains only 1–10 % of parameters
- 📦 **GGUF export** – quantise & push to Hugging Face Hub in one step
- 🌐 **FastAPI** REST endpoint for easy integration
- 🖥️ **Tkinter** desktop GUI for local interactive use

---

## 🏗️ Model & Training Details

### Base Model

| Parameter | Value |
|---|---|
| Base model | `unsloth/llama-3-8b-bnb-4bit` |
| Max sequence length | 2048 |
| Load in 4-bit | ✅ |
| dtype | Auto-detect (BFloat16 / Float16) |

### LoRA Configuration

| Parameter | Value |
|---|---|
| Rank `r` | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0 (optimised) |
| Bias | none (optimised) |
| Gradient checkpointing | `"unsloth"` (saves ~30 % VRAM) |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |

### Training Hyperparameters

| Parameter | Value |
|---|---|
| Batch size per device | 3 |
| Gradient accumulation steps | 6 |
| Effective batch size | 18 |
| Max steps | 120 |
| Warmup steps | 5 |
| Learning rate | 2e-4 |
| LR scheduler | Linear |
| Optimiser | AdamW 8-bit |
| Weight decay | 0.01 |
| Seed | 3407 |

---

## 📊 Dataset

The model is trained on the custom **[odenmehmet/TRObjectTEst](https://huggingface.co/datasets/odenmehmet/TRObjectTEst)** dataset hosted on Hugging Face.

The dataset follows the **Alpaca** instruction-following format:

```
Below is an input that provides context. Write a response that appropriately completes the request.

### Input:
{instruction}

### Response:
{output}
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- NVIDIA GPU (Tesla T4, V100, A100, or equivalent)
- [Google Colab](https://colab.research.google.com/) (recommended) or a local CUDA environment
- A [Hugging Face](https://huggingface.co/) account and API token

### Installation

```bash
# Install Unsloth and Flash Attention
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```

For the deployment notebooks, also install:

```bash
pip install fastapi uvicorn llama-cpp-python nest_asyncio
```

---

## 🧑‍💻 Usage

### 1. Fine-Tuning

Open and run **`Llama-3 8b Unsloth Fine-tuning_Mehmet ÖDEN.ipynb`** step by step:

**Step 1 – Login to Hugging Face**
```python
!huggingface-cli login
```

**Step 2 – Load the base model with 4-bit quantisation**
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)
```

**Step 3 – Apply LoRA adapters**
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
```

**Step 4 – Train with SFTTrainer**
```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=3,
        gradient_accumulation_steps=6,
        max_steps=120,
        learning_rate=2e-4,
        output_dir="outputs",
    ),
)
trainer.train()
```

**Step 5 – Export to GGUF and push to Hugging Face Hub**
```python
model.push_to_hub_gguf(
    "your-hf-username/your-model",
    tokenizer,
    quantization_method="q4_k_m",
    token="HF_TOKEN",
)
```

### 2. Inference (in notebook)

```python
FastLanguageModel.for_inference(model)

inputs = tokenizer(
    [alpaca_prompt.format("Bir form oluştur ve bir panel ekle.", "")],
    return_tensors="pt",
).to("cuda")

from transformers import TextStreamer
_ = model.generate(**inputs, streamer=TextStreamer(tokenizer), max_new_tokens=128)
```

---

## 🌐 FastAPI Deployment

Open **`TRObjectLlama_Mehmet ÖDEN.ipynb`** and run the FastAPI section.
The server starts on `http://0.0.0.0:8000`.

**Endpoint:** `POST /generate`

```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Bir form oluştur ve içine bir buton ekle.", "max_length": 200}'
```

**Response:**

```json
{
  "generated_text": "..."
}
```

> **Note:** Update `model_path` in the notebook to point to your locally saved GGUF file before starting the server.

---

## 🖥️ Tkinter Desktop App

For a local, GUI-based experience, run the **Tkinter** section of `TRObjectLlama_Mehmet ÖDEN.ipynb`.
The app loads the GGUF model from a local path and lets you type prompts and view generated TRObject code in real time.

```python
model_path = "D:/Model/unsloth.Q4_K_M.gguf"   # update to your local path
```

---

## 📸 Screenshots

<div align="center">
  <img src="Screenshot 2024-08-15 112742.png" alt="Training screenshot 1" width="300">
  <img src="Screenshot 2024-08-15 113014.png" alt="Training screenshot 2" width="300">
  <img src="Screenshot 2024-08-15 113045.png" alt="Training screenshot 3" width="300">
  <img src="Screenshot 2024-08-15 113231.png" alt="Training screenshot 4" width="300">
  <img src="Screenshot 2024-08-15 113255.png" alt="Training screenshot 5" width="300">
  <img src="Screenshot 2024-08-15 113331.png" alt="Training screenshot 6" width="300">
</div>

---

## 📁 Repository Structure

```
Llama-3-8b-Unsloth-Fine-tuning/
├── Llama-3 8b Unsloth Fine-tuning_Mehmet ÖDEN.ipynb   # Fine-tuning notebook
├── TRObjectLlama_Mehmet ÖDEN.ipynb                     # Deployment notebook
├── Llama-3 8b Unsloth Fine-tuning_Mehmet ÖDEN.pdf      # PDF report
├── Screenshot 2024-08-15 *.png                         # Training screenshots
└── README.md
```

---

## 🔗 Resources

- 📓 [Fine-tuning Notebook (PDF)](https://github.com/odenmehmet/Llama-3-8b-Unsloth-Fine-tuning/blob/main/Llama-3%208b%20Unsloth%20Fine-tuning_Mehmet%20%C3%96DEN.pdf)
- 🤗 [Unsloth on GitHub](https://github.com/unslothai/unsloth)
- 🤗 [Llama-3 on Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- 📦 [TRL SFTTrainer Docs](https://huggingface.co/docs/trl/sft_trainer)
- 📊 [Training Dataset](https://huggingface.co/datasets/odenmehmet/TRObjectTEst)

---

## 👤 Author

**Mehmet ÖDEN**
Fine-tuning & deployment of Llama-3 8B for Turkish TRObject code generation.

---

## �� License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).
