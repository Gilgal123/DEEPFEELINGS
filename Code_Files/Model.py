# imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig
import os

# Load TinyLLaMA
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Apply LoRA
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS",
)

class TinyLlamaClassifier(nn.Module):
    """
    @brief Wrapper model adding a classification head on top of TinyLLaMA base model.

    @details
    Wraps TinyLLaMA transformer with a linear classification layer.
    Supports freezing the base model parameters to train only the classification head.
    Computes cross-entropy loss if labels are provided.

    @param num_labels (int) Number of emotion classes to classify.
    @param tinyllama (nn.Module) Pretrained TinyLLaMA model.
    @param freeze_base (bool) If True, freezes the TinyLLaMA base model parameters.
    """
    def __init__(self, num_labels, tinyllama, freeze_base=False):
        super().__init__()
        self.tinyllama = tinyllama
        
        if freeze_base:
            for param in self.tinyllama.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(tinyllama.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        @brief Forward pass of the model.

        @param input_ids (torch.Tensor) Token IDs tensor of shape (batch_size, seq_len).
        @param attention_mask (torch.Tensor, optional) Attention mask tensor.
        @param labels (torch.Tensor, optional) Ground truth labels for computing loss.

        @return dict with:
            - "loss" (torch.Tensor or None): Cross-entropy loss if labels provided, else None.
            - "logits" (torch.Tensor): Classification logits of shape (batch_size, num_labels).
        """
        outputs = self.tinyllama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )
        last_token_hidden = outputs.hidden_states[-1][:, -1, :]
        logits = self.classifier(last_token_hidden)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}

    def save_full_model(self, path = "./checkpoint.pt"):
        """
        @brief Saves the model state dict (LoRA parameters + classifier) to disk.

        @param path (str) File path to save the checkpoint.
        """
        torch.save(self.state_dict(), path)
        print(f"Saved full model (Only LoRA and classifier) to: {path}")

def load_or_init_model(num_labels, weights_path = "./checkpoint.pt") -> TinyLlamaClassifier:
    """
    @brief Load TinyLLaMA model with LoRA and classification head from checkpoint if available,
           else initialize new model.

    @param num_labels (int) Number of classification labels.
    @param weights_path (str) Path to the checkpoint file.

    @return TinyLlamaClassifier model instance.
    """
    print(f"Loading base model '{model_id}' with LoRA and classifier...")

    base_model = AutoModelForCausalLM.from_pretrained(model_id)

    lora_model = get_peft_model(base_model, lora_config)
    model = TinyLlamaClassifier(num_labels, lora_model)

    if os.path.exists(weights_path):
        print(f"Checkpoint found at '{weights_path}'. Loading weights...")
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
        print("Weights loaded successfully.")
    else:
        print(f"No checkpoint found at '{weights_path}'. Initialized new model.")

    # Print number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    return model

def load_or_init_model_no_lora(num_labels, weights_path="./checkpoint.pt") -> TinyLlamaClassifier:
    """
    @brief Load TinyLLaMA base model with classification head (no LoRA) from checkpoint if available,
           else initialize new model with frozen base parameters.

    @param num_labels (int) Number of classification labels.
    @param weights_path (str) Path to the checkpoint file.

    @return TinyLlamaClassifier model instance.
    """
    print(f"Loading base model '{model_id}' without LoRA, with classifier...")

    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    model = TinyLlamaClassifier(num_labels, base_model, freeze_base=True)

    if os.path.exists(weights_path):
        print(f"Checkpoint found at '{weights_path}'. Loading weights...")
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
        print("Weights loaded successfully.")
    else:
        print(f"No checkpoint found at '{weights_path}'. Initialized new model.")

    # Print number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    return model
