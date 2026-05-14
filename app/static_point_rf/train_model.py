import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

print("\n" + "="*50)
print("ΕΚΠΑΙΔΕΥΣΗ ΜΟΝΤΕΛΟΥ RANDOM FOREST")
print("="*50)

# 1. Φόρτωση των δεδομένων από το Excel/CSV
print("[1/4] Φόρτωση δεδομένων από το 'final_merged_dataset_relative.csv'...")
df = pd.read_csv('final_merged_dataset_relative.csv', encoding='utf-8-sig')

# 2. Διαχωρισμός: Τι ψάχνουμε (Γράμμα) και τι έχουμε (Συντεταγμένες)
# Το 'y' είναι η στήλη 'label' (τα γράμματα Α, Β, G)
y = df['label']
# Το 'X' είναι όλα τα υπόλοιπα (οι 42 συντεταγμένες των σημείων)
X = df.drop('label', axis=1)

# Κρατάμε το 80% των δεδομένων για διάβασμα (Εκπαίδευση) 
# και το 20% για διαγώνισμα (Τεστ) ώστε να δούμε αν τα έμαθε καλά!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Δημιουργία και Εκπαίδευση του Random Forest (Με 100 "δέντρα αποφάσεων")
print("[2/4] Ξεκινάει η εκπαίδευση (μπορεί να πάρει λίγα δευτερόλεπτα)...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Το Διαγώνισμα (Τεστ)
print("[3/4] Βαθμολόγηση του μοντέλου...")
y_pred = model.predict(X_test) # Του δίνουμε τα δεδομένα τεστ (χωρίς τα γράμματα) να μαντέψει
score = accuracy_score(y_test, y_pred) # Συγκρίνουμε τις μαντεψιές του με τα πραγματικά γράμματα

print(f"\n---> ΑΚΡΙΒΕΙΑ (ACCURACY): {score * 100:.2f}% <---")

# 5. Αποθήκευση του "εγκεφάλου"
print("\n[4/4] Αποθήκευση του εκπαιδευμένου μοντέλου...")
with open('rf_model_merged_relative.pkl', 'wb') as f:
    pickle.dump(model, f)

print("ΤΕΛΟΣ! Το αρχείο 'rf_model_merged_relative.pkl' δημιουργήθηκε επιτυχώς.")
print("="*50 + "\n")