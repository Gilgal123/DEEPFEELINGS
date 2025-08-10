import sys
import torch
from torch.utils.data import DataLoader
from Model import load_or_init_model, tokenizer
from DataSet import get_cached_deepfeeling_dataset

DEVICE = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    """
    @brief Prepares a batch for model input by tokenizing texts and encoding labels.
    
    @param batch List of dataset samples, each sample is a dict with 'text' and 'reduced_label'.
    @return dict Dictionary containing input_ids, attention_mask, and labels tensors.
    """
    texts = [x['text'] for x in batch]
    labels = torch.tensor([label_names.index(x['reduced_label']) for x in batch], dtype=torch.long)
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels
    }

def evaluate_model(checkpoint_path="./checkpoint"):
    """
    @brief Loads the model from the given checkpoint path and evaluates on the test dataset.
    
    @param checkpoint_path Path to the model checkpoint file. Default is "./checkpoint".
    @return None Prints accuracy on the test set.
    """
    print(f"Loading dataset...")
    dataset, class_weights, label_names = get_cached_deepfeeling_dataset(
        force_reload=False, prompt_format=False, only_low_class_count=True
    )

    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = load_or_init_model(len(label_names), weights_path=checkpoint_path)
    model.to(DEVICE)
    model.eval()

    test_loader = DataLoader(dataset['test'], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs['logits'].argmax(dim=-1)

            correct += (preds == labels).sum().item()
            total += input_ids.size(0)

    accuracy = correct / total
    print(f"Test set accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    """
    @brief Main entry point. Reads checkpoint path from command line or prompts user, 
           then runs model evaluation.
    
    Usage:
        python main.py [checkpoint_path]
    """
    if len(sys.argv) > 1:
        checkpoint = sys.argv[1]
    else:
        checkpoint = input("Enter checkpoint path (default: ./checkpoint): ").strip() or "./checkpoint"

    evaluate_model(checkpoint)
