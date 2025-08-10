# Emotion Classification Model Checkpoints

This repository contains six trained checkpoints of TinyLLaMA-based models for text emotion classification. Each checkpoint corresponds to a different experimental setup exploring the effects of LoRA fine-tuning, input prompt formatting, and dataset selection on model performance.

---

## Checkpoints Overview

1. **Linear Head Only, Raw Input**  
   - Model with only a linear classification head.  
   - Input sentences are raw, with no prompt added.  
   - No LoRA applied.

2. **Linear Head + LoRA, Raw Input**  
   - Linear classification head combined with LoRA fine-tuning.  
   - Input sentences remain raw, without prompt.

3. **Linear Head Only, Prompted Input**  
   - Linear head without LoRA.  
   - Input sentences prefixed with the prompt:  
     ```
     Classify the emotion of the following sentence:
     Sentence:
     ```

4. **Linear Head + LoRA, Prompted Input**  
   - Linear head with LoRA.  
   - Input sentences prefixed as in checkpoint (3).

5. **Linear Head + LoRA (Reduced Rank, Expanded Targets), Raw Input**  
   - LoRA rank reduced by half to create a lighter adaptation.  
   - LoRA applied across more attention projection matrices: query, value, key, and output projections.  
   - Input sentences are raw, without prompt.

6. **Linear Head + LoRA (Same as #5), Reduced Dataset**  
   - Same LoRA configuration as checkpoint (5).  
   - Trained only on the smaller, cleaner "Emotions" dataset instead of merged datasets.  
   - Input sentences raw, without prompt.

---

## Important Note on Checkpoint Sizes

The checkpoints are very large (several gigabytes each). To keep this repository manageable, **only the final best-performing checkpoint (checkpoint 6)** is included here via a download link.

If you need access to other checkpoints, please contact us or check separate provided links.

---

## Summary

These experiments highlight the impact of fine-tuning techniques, input prompt engineering, and dataset choice on emotion classification accuracy. Interestingly, training on a focused, smaller dataset with optimized LoRA configurations achieved superior results compared to larger but noisier datasets.

---

## Usage

Download the desired checkpoint and use the provided training and evaluation scripts to reproduce results or serve as baselines for further work.

---

## Contact

For questions or requests regarding other checkpoints or usage, please open an issue or contact the repository maintainers.

