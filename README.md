# 💬 DEEPFEELINGS

> Emotion Classification from Text Using TinyLLaMA + LoRA

**DEEPFEELINGS** is a compact, efficient model for classifying emotions in text.  
It is built on top of **TinyLLaMA** and fine-tuned using **LoRA (Low-Rank Adaptation)**, allowing for fast inference and low resource usage.

---

## 🔥 Features

- ✅ Lightweight: Built on TinyLLaMA for speed and size
- 🎯 Emotion detection in sentences and short texts
- 🧠 Trained using LoRA with multiple strategies:
  - Linear head on frozen LLM
  - Prompt-based classification
  - LoRA-finetuned + classifier
- 🏷 7 emotion categories:
  - `Anger`, `Fear`, `Joy`, `Sadness`, `Love`, `Surprise`, `Neutral`

---

## 📂 Dataset

DEEPFEELINGS uses a combined and relabeled dataset from:
- [GoEmotions](https://github.com/google-research/goemotions)
- [Emotions Dataset (dair-ai)](https://github.com/dair-ai/emotion_dataset)

These are mapped into a simplified label set of 7 universal emotions.

---

## 🚀 Quickstart

```bash
git clone https://github.com/Gilgal123/deepfeelings.git
cd deepfeelings

you can train your own model using the train.py module.

for evaluation run:
  python ./eval.py [path to checkpoint] 

