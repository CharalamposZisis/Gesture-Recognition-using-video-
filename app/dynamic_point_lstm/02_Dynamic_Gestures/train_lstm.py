import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# --- 1. ΦΟΡΤΩΣΗ ΚΑΙ ΠΡΟΕΤΟΙΜΑΣΙΑ ΔΕΔΟΜΕΝΩΝ ---
print("Φόρτωση δεδομένων...")
df = pd.read_csv('two_hands_sequences.csv')

# Μετατρέπουμε τα labels (1, 2, 3) σε (0, 1, 2) 
# Πλέον το IDLE (3) θα γίνει 2 εσωτερικά για την AI
y = df['label'].values - 1 
X = df.drop('label', axis=1).values

# Reshape: (Αριθμός δειγμάτων, 30 καρέ, 84 συντεταγμένες)
X = X.reshape(-1, 30, 84)

# Χωρίζουμε σε Train (80%) και Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Μετατροπή σε PyTorch Tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=8, shuffle=True)

# --- 2. ΟΡΙΣΜΟΣ ΤΟΥ ΜΟΝΤΕΛΟΥ LSTM ---
class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GestureLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) 
        return out

# --- ΕΔΩ ΕΙΝΑΙ ΟΙ ΑΛΛΑΓΕΣ ---
num_classes = 3  # ΑΛΛΑΓΗ ΑΠΟ 2 ΣΕ 3
model = GestureLSTM(input_size=84, hidden_size=64, num_layers=2, num_classes=num_classes)
# ----------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 3. ΕΚΠΑΙΔΕΥΣΗ (TRAINING) ---
print("Ξεκινάει η εκπαίδευση για 3 κλάσεις...")
epochs = 40
for epoch in range(epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# --- 4. ΕΛΕΓΧΟΣ ΑΚΡΙΒΕΙΑΣ ---
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f'\nΑκρίβεια Μοντέλου: {accuracy * 100:.2f}%')

# --- 5. ΑΠΟΘΗΚΕΥΣΗ ---
torch.save(model.state_dict(), 'gesture_lstm.pth')
print("Το αναβαθμισμένο μοντέλο αποθηκεύτηκε!")