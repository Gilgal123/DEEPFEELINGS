import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt
from Model import load_or_init_model, load_or_init_model_no_lora, tokenizer
from DataSet import get_cached_deepfeeling_dataset
import time
from datetime import datetime

# === Config ===
BATCH_SIZE = 4
EPOCHS = 20
LR = 2e-4
DEVICE = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda" if torch.cuda.is_available() else "cpu")

print(DEVICE)
# === Load dataset and model ===
dataset, class_weights, label_names = get_cached_deepfeeling_dataset(force_reload=True, prompt_format = False, only_low_class_count = True)
num_labels = len(label_names)
train_size = len(dataset['train'])
num_batches = train_size / BATCH_SIZE

print(f"Loaded dataset with {train_size} training samples.")
print(f"Classes: {label_names}")

"""
Insert current model to train
load_or_init_model_no_lora for tinyllama without lora
load_or_init_model for tinyllama with lora
"""
model = load_or_init_model(num_labels)
#model = load_or_init_model_no_lora(num_labels)
model.to(DEVICE)
model.train()

# === Collate function ===
def collate_fn(batch):
    """
    @brief Collate function to prepare a batch for training.
    
    @param batch List of dataset samples.
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

train_loader = DataLoader(dataset['train'], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(dataset['validation'], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# === Optimizer & Loss with class weights ===
optimizer = AdamW(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

# === For live plotting ===
losses, accuracies, train_accuracies = [2], [], [0.14]
best_val_acc = 0.0
# === Training loop ===
print("Checking the accuracy before the training.")
#Checking the accuracy before the training.
model.eval()
val_correct = 0
val_samples = 0
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs['logits'].argmax(dim=-1)
        val_correct += (preds == labels).sum().item()
        val_samples += input_ids.size(0)
        #break #TODO remove.
val_acc = val_correct / val_samples
print(f"Epoch 0/{EPOCHS} - Val Acc: {val_acc:.4f}")
accuracies.append(val_acc)

for epoch in range(EPOCHS):
    left_batchs = num_batches
    print(f"epoch loop {epoch}")
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch_idx, batch in enumerate(train_loader):
        
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        if batch_idx == 0:
            start = time.time()
        if (batch_idx % 500) == 0:
            end = time.time()
            left_batchs = left_batchs - 500
            time_for_500_batch = end - start
            start = end
            print(f"Time for 500 batchs is: {time_for_500_batch}, with batch size: {BATCH_SIZE}")
            print(f"{left_batchs} batches left for epoch")
            print(f"Estimated time (hours) left for current epoch: {((left_batchs/500) * time_for_500_batch/3600):.3f}")

        if (batch_idx % 100) ==0:
            print(f"Batch {batch_idx}: loss = {loss.item():.4f}")
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)

        preds = outputs['logits'].argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += input_ids.size(0)
        #break #TODO remove.

    avg_loss = total_loss / total_samples
    train_acc = total_correct / total_samples

    # Validation
    model.eval()
    val_correct = 0
    val_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs['logits'].argmax(dim=-1)
            val_correct += (preds == labels).sum().item()
            val_samples += input_ids.size(0)
            #break #TODO remove

    val_acc = val_correct / val_samples
    model.save_full_model("./checkpoint.pt")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_full_model("./checkpoint_best.pt")
        print(f"New best model saved with val acc: {val_acc:.4f}")

    # Update plot data
    losses.append(avg_loss)
    accuracies.append(val_acc)
    train_accuracies.append(train_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

    log_line = f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - {log_line}\n"

    with open("training_log.txt", "a") as f:
        f.write(log_entry)

#Plot results
fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

epochs = range(EPOCHS + 1)
ax_loss.plot(epochs, losses, label="Train Loss", color="blue")
ax_loss.set_title("Training Loss")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss")
ax_loss.legend()

# Accuracy plot (both train and validation)
ax_acc.plot(epochs, train_accuracies, label="Train Accuracy", color="green")
ax_acc.plot(epochs, accuracies, label="Validation Accuracy", color="orange")
ax_acc.set_title("Accuracy")
ax_acc.set_xlabel("Epoch")
ax_acc.set_ylabel("Accuracy")
ax_acc.set_ylim(0, 1)
ax_acc.legend()
plt.tight_layout()
plt.show()

print("Training complete.")

