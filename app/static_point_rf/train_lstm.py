import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

print("\n" + "="*55)
print("  ΕΚΠΑΙΔΕΥΣΗ ΔΥΝΑΜΙΚΟΥ ΜΟΝΤΕΛΟΥ (LSTM) ΓΙΑ ΦΡΑΣΕΙΣ")
print("="*55)

# 1. Φόρτωση δεδομένων (Αυτόματη εύρεση του CSV)
csv_file = 'sign_language_phrases.csv'
if not os.path.isfile(csv_file):
    csv_file = 'sign_language_phrases.csv'

print(f"[1/5] Φόρτωση δεδομένων από το '{csv_file}'...")
df = pd.read_csv(csv_file)

# 2. Διαχωρισμός: Τι ψάχνουμε (Labels) και τι έχουμε (Χρονική Ακολουθία)
# Τα labels μας είναι 1, 2, 3, 4, 5. Το νευρωνικό θέλει να ξεκινάνε από 0 (0, 1, 2, 3, 4).
y = df['label'].astype(int) - 1 

# Μετατροπή των labels σε One-Hot Encoding (π.χ. το 0 γίνεται [1, 0, 0, 0, 0])
y = to_categorical(y, num_classes=5) 

# Όλες οι συντεταγμένες (5.880 νούμερα ανά γραμμή)
X_flat = df.drop('label', axis=1).values

# 3. Το "Κόλπο του LSTM" (Reshaping)
# Το LSTM θέλει τα δεδομένα σε 3D μορφή: (Αριθμός Βίντεο, Καρέ ανά Βίντεο, Σημεία ανά Καρέ)
# Ξέρουμε ότι έχουμε 70 καρέ x 84 σημεία (σύνολο 5880)
print("[2/5] Μετατροπή δεδομένων από Flat σε 3D...")
X = X_flat.reshape(-1, 70, 84)

# Κρατάμε το 80% για διάβασμα και το 20% για διαγώνισμα
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Χτίσιμο της αρχιτεκτονικής του LSTM
print("[3/5] Δημιουργία της αρχιτεκτονικής του Δικτύου...")
model = Sequential()
# Πρώτο επίπεδο LSTM: Διαβάζει τα καρέ με τη σειρά
model.add(LSTM(64, return_sequences=True, input_shape=(70, 84)))
model.add(Dropout(0.2)) # Προστασία από "παπαγαλία" (overfitting)

# Δεύτερο επίπεδο LSTM: Συνοψίζει τη συνολική κίνηση
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.2))

# Κλασικοί νευρώνες (Dense) για την τελική απόφαση
model.add(Dense(64, activation='relu'))
model.add(Dense(5, activation='softmax')) # 5 νευρώνες στην έξοδο, ένας για κάθε φράση

# Ρυθμίσεις του "Δασκάλου" (Optimizer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Εκπαίδευση
print("\n[4/5] Ξεκινάει η προπόνηση! \n(Κάθε 'Epoch' είναι ένα πέρασμα όλης της ύλης)\n")
# epochs=50 σημαίνει ότι θα διαβάσει τα δεδομένα 50 φορές
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

# Αξιολόγηση (Πώς τα πήγε στο 20% που δεν είχε δει ποτέ;)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n---> ΑΚΡΙΒΕΙΑ ΣΤΟ ΔΙΑΓΩΝΙΣΜΑ (TEST ACCURACY): {accuracy * 100:.2f}% <---")

# 6. Αποθήκευση
print("\n[5/5] Αποθήκευση του εκπαιδευμένου μοντέλου...")
model.save('lstm_phrase_model.h5')

print("ΤΕΛΟΣ! Το αρχείο 'lstm_phrase_model.h5' δημιουργήθηκε επιτυχώς.")
print("="*55 + "\n")