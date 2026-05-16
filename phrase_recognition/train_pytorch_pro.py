import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import copy
import os

print("\n" + "="*65)
print("  ΕΚΠΑΙΔΕΥΣΗ PyTorch LSTM (PRO EDITION) ΓΙΑ ΝΟΗΜΑΤΙΚΗ")
print("="*65)

# --- 1. ΑΥΤΟΜΑΤΗ ΕΠΙΛΟΓΗ ΥΛΙΚΟΥ (GPU ή CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Σύστημα Εκπαίδευσης: {device.type.upper()}")

# --- 2. ΟΡΙΣΜΟΣ ΜΟΝΤΕΛΟΥ ---
class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GestureLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# --- 3. ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ & DATALOADERS ---
csv_file = 'master_phrases_dataset.csv'
print(f"[*] Φόρτωση δεδομένων από '{csv_file}'...")
df = pd.read_csv(csv_file)

y = df['label'].astype(int).values - 1
X_flat = df.drop('label', axis=1).values
X = X_flat.reshape(-1, 70, 84)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Μετατροπή σε Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# Δημιουργία DataLoaders (Πακετάκια των 16 βίντεο)
batch_size = 16
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_test_t, y_test_t)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# --- 4. SETUP ΕΚΠΑΙΔΕΥΣΗΣ ---
input_size = 84
hidden_size = 128 # Αυξήσαμε λίγο τη "μνήμη" του δικτύου
num_layers = 2
num_classes = 5 

model = GestureLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Scheduler (Μειώνει το learning rate αν δεν βλέπει βελτίωση)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# --- 5. EARLY STOPPING SETUP ---
epochs = 200
patience = 15 # Πόσες εποχές θα περιμένει πριν τα παρατήσει
patience_counter = 0
best_val_loss = float('inf')
best_model_weights = copy.deepcopy(model.state_dict())

# Ιστορικό για τα γραφήματα
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

print("\n[*] Ξεκινάει η προπόνηση (Με Early Stopping & Scheduler)...\n")

for epoch in range(epochs):
    # --- ΦΑΣΗ ΔΙΑΒΑΣΜΑΤΟΣ (TRAINING) ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / total

    # --- ΦΑΣΗ ΔΙΑΓΩΝΙΣΜΑΤΟΣ (VALIDATION) ---
    model.eval()
    val_running_loss, val_correct, val_total = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    val_loss = val_running_loss / len(val_loader.dataset)
    val_acc = val_correct / val_total
    
    # Ενημέρωση ιστορικού
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    print(f"Epoch [{epoch+1:03d}/{epochs:03d}] | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%")

    # Ενημέρωση Scheduler
    scheduler.step(val_loss)

    # --- EARLY STOPPING LOGIC ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n[!] Early Stopping: Δεν υπήρξε βελτίωση για {patience} εποχές. Η εκπαίδευση σταματάει.")
            break

# --- 6. ΑΠΟΘΗΚΕΥΣΗ & ΓΡΑΦΗΜΑΤΑ ---
print("\n[*] Αποθήκευση της ΚΑΛΥΤΕΡΗΣ έκδοσης του μοντέλου...")
model.load_state_dict(best_model_weights)
torch.save(model.state_dict(), 'best_pytorch_phrase_model.pth')
print(" -> Το αρχείο 'best_pytorch_phrase_model.pth' είναι έτοιμο!")

# Ζωγραφίζουμε την πορεία
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss Progression')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Accuracy Progression')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

print("="*65 + "\n")