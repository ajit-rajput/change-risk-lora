# Change Risk Assessment using LoRA

This POC project demonstrates how **Low-Rank Adaptation (LoRA)** can be used to teach a language model **operational risk judgment** for **security and access control changes**, using a **fully local setup**.

The goal is not to predict vulnerabilities, but to **reason about change impact before execution**, similar to how senior engineers review changes during approval workflows.

## 🧠 What This Project Does

Given a natural-language change description, the model produces a **structured risk assessment**:

```json
{
  "risk_level": "High",
  "risk_factors": [],
  "blast_radius": "Account",
  "rollback_complexity": "Easy",
  "recommended_action": "",
  "confidence": "High"
}
```
---

## Dataset

The training dataset consists of **hand-crafted gold samples** representing
realistic security and access-control change scenarios.

Characteristics:
- Small, high-signal dataset
- Synthetic but production-inspired
- Focused on risk amplification patterns
- Assumes a production environment by default

The dataset is intentionally small to demonstrate how LoRA can encode
domain judgment with limited data.

## End to End Flow

Dataset (JSON)  
→ LoRA Training (local, CPU)  
→ LoRA Adapter Weights  
→ Merged Model  
→ GGUF Export  
→ Ollama (local inference)

---

## Why LoRA

LoRA is a good fit because:
	•	Risk assessment is pattern-based, not factual recall
	•	Organizational judgment is consistent but subtle
	•	Small datasets can encode strong behavioral bias
	•	Only a few million parameters need to be trained

This project fine-tunes behavior, not knowledge.

## Ollama Integration
	•	Ollama is used only for inference
	•	The LoRA adapter is merged into the base model
	•	The merged model is converted to GGUF
	•	The final model runs locally via Ollama

Note: No base model needs to be preloaded in Ollama; the merged GGUF model in further steps is fully self-contained.

---
## Steps
### LoRA Training
```bash
python training/train_lora.py
```

### Merge LoRA Adapter into Base Model
```bash
python training/merge_lora.py

```
### Export Merged Model to GGUF
```
git clone https://github.com/ggerganov/llama.cpp.git
pip install sentencepiece protobuf

python llama.cpp/convert_hf_to_gguf.py \
  training/merged_model \
  --outfile change-risk-lora.gguf
```
## Inference

The LoRA-adapted model is exported to **GGUF** format and executed locally using **Ollama**.

Due to strict parsing and validation behavior in Ollama v0.13.x, the model is registered using a minimal `Modelfile` that directly references the merged GGUF artifact.

### Model Registration

From the project root directory:

```bash
ollama create changeriskmodel -f Modelfile
```

```bash
ollama list
```

```bash
ollama run changeriskmodel
```

Example Prompt

```text
Assess the security risk of the change:
Disable MFA for admin role to simplify emergency access
```
Respond strictly in JSON.

```json
{
  "risk_level": "Critical",
  "risk_factors": [
    "Disabling MFA weakens administrator account security",
    "Increases likelihood of unauthorized access"
  ],
  "blast_radius": "Organization-wide",
  "rollback_complexity": "Easy",
  "recommended_action": "Use a monitored break-glass account instead of disabling MFA",
  "confidence": "High"
}