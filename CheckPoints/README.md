# Emotion Classification Model Checkpoints

This directory contains six trained checkpoints of TinyLLaMA-based models for text emotion classification, each representing a different experimental configuration. The goal was to explore the effects of LoRA fine-tuning, input prompt design, and dataset choice on model performance.

## Checkpoints Description

1. **Linear Head Only, Raw Input**  
   - Model with only a linear classification head.  
   - Input sentences are raw, without any prompt modification.  
   - No LoRA applied.

2. **Linear Head + LoRA, Raw Input**  
   - Linear head combined with LoRA fine-tuning.  
   - Input sentences remain raw, without any prompt.

3. **Linear Head Only, Prompted Input**  
   - Linear head without LoRA.  
   - Input sentences are prefixed with:  
     ```
     Classify the emotion of the following sentence:
     Sentence:
     ```

4. **Linear Head + LoRA, Prompted Input**  
   - Linear head with LoRA.  
   - Input sentences prefixed as in (3).

5. **Linear Head + LoRA (Reduced Rank, Expanded Targets), Raw Input**  
   - LoRA rank reduced by half for lighter adaptation.  
   - LoRA applied across more attention projection matrices (query, value, key, and output).  
   - Input sentences raw, without prompt.

6. **Linear Head + LoRA (Same as #5), Reduced Dataset**  
   - Same LoRA configuration as checkpoint (5).  
   - Trained only on the smaller, cleaner "Emotions" dataset instead of merged datasets.  
   - Input sentences raw, without prompt.

## Summary

These checkpoints illustrate how modifications to fine-tuning approach, input formatting, and dataset composition influence classification effectiveness. Notably, reducing dataset size to a focused subset (checkpoint 6) yielded better results despite fewer training samples, emphasizing quality over quantity.

## Usage

Load these checkpoints to reproduce experiments or serve as baselines for further development. Refer to training and evaluation scripts for integration.
