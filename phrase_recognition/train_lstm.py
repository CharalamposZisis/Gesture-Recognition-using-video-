import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

print("\n" + "="*60)
print("  ΕΚΠΑΙΔΕΥΣΗ ΔΥΝΑΜΙΚΟΥ ΜΟΝΤΕΛΟΥ (LSTM) - PRO VERSION")
print("="*60)

# 1. Φόρτωση δεδομένων
csv_file = 'phrase_recognition/master_phrases_dataset.csv'
if not os.path.isfile(csv_file):
    csv_file = 'master_phrases_dataset.csv'

print(f"[1/6] Φόρτωση δεδομένων από το '{csv_file}'...")
df = pd.read_csv(csv_file)

# 2. Διαχωρισμός 
y = df['label'].astype(int) - 1 
y = to_categorical(y, num_classes=5) 
X_flat = df.drop('label', axis=1).values

# 3. Reshaping σε 3D
print("[2/6] Μετατροπή δεδομένων σε 3D (Videos x Frames x Points)...")
X = X_flat.reshape(-1, 70, 84)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Αρχιτεκτονική LSTM
print("[3/6] Στήσιμο Νευρωνικού Δικτύου...")
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(70, 84)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Ρύθμιση των "Έξυπνων" Αισθητήρων (Callbacks)
print("[4/6] Ενεργοποίηση Early Stopping & Checkpoints...")

# Σταματάει αν το validation loss δεν βελτιωθεί για 15 συνεχόμενες εποχές
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

# Αποθηκεύει ΣΥΝΕΧΕΙΑ την καλύτερη έκδοση
checkpoint = ModelCheckpoint('best_lstm_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# 6. Εκπαίδευση
print("\n[5/6] Ξεκινάει η προπόνηση (Θα σταματήσει αυτόματα στο απόγειό του)...\n")
history = model.fit(
    X_train, y_train, 
    epochs=200,          # Βάλαμε μεγάλο νούμερο, αλλά θα κοπεί νωρίτερα
    batch_size=8, 
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint] # Εδώ μπαίνουν οι αισθητήρες!
)

# 7. Αξιολόγηση και Γραφήματα
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n---> ΤΕΛΙΚΗ ΑΚΡΙΒΕΙΑ ΣΤΟ ΔΙΑΓΩΝΙΣΜΑ (TEST ACCURACY): {accuracy * 100:.2f}% <---")
print("\n[6/6] Δημιουργία γραφημάτων...")

# --- Ζωγραφίζουμε την πορεία της εκπαίδευσης ---
plt.figure(figsize=(12, 5))

# Γράφημα για το Loss (Το Λάθος)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Διάβασμα (Train Loss)')
plt.plot(history.history['val_loss'], label='Διαγώνισμα (Val Loss)')
plt.title('Πορεία Λάθους (Loss)')
plt.xlabel('Εποχές (Epochs)')
plt.ylabel('Λάθος')
plt.legend()

# Γράφημα για την Ακρίβεια (Accuracy)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Διάβασμα (Train Acc)')
plt.plot(history.history['val_accuracy'], label='Διαγώνισμα (Val Acc)')
plt.title('Πορεία Ακρίβειας (Accuracy)')
plt.xlabel('Εποχές (Epochs)')
plt.ylabel('Ακρίβεια')
plt.legend()

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("ΤΕΛΟΣ! Η καλύτερη έκδοση του μοντέλου σώθηκε ως 'best_lstm_model.h5'.")
print("="*60 + "\n")