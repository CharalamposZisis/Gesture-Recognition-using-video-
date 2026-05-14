import pandas as pd
import os

print("\n" + "="*50)
print("  ΕΡΓΑΛΕΙΟ ΜΕΤΑΤΡΟΠΗΣ ΣΕ ΣΧΕΤΙΚΕΣ ΣΥΝΤΕΤΑΓΜΕΝΕΣ")
print("="*50)

old_file = 'greek_hand_dataset.csv'
new_file = 'greek_hand_dataset_relative.csv'

if not os.path.isfile(old_file):
    print(f"Σφάλμα: Δεν βρέθηκε το αρχείο {old_file}.")
else:
    try:
        print("[1/3] Φόρτωση του παλιού dataset...")
        # Διαβάζουμε το αρχείο με τα απόλυτα νούμερα
        df = pd.read_csv(old_file, encoding='windows-1253')
        
        print("[2/3] Υπολογισμός των νέων (σχετικών) συντεταγμένων...")
        # Δημιουργούμε ένα αντίγραφο για να βάλουμε τα νέα νούμερα
        df_relative = df.copy()
        
        # Κρατάμε τις στήλες του καρπού (x0 και y0)
        wrist_x = df['x0']
        wrist_y = df['y0']
        
        # Αφαιρούμε τον καρπό από ΚΑΘΕ σημείο (από το 0 έως το 20)
        for i in range(21):
            df_relative[f'x{i}'] = df[f'x{i}'] - wrist_x
            df_relative[f'y{i}'] = df[f'y{i}'] - wrist_y
            
        print("[3/3] Αποθήκευση του νέου αρχείου...")
        # Αποθηκεύουμε το νέο, "έξυπνο" dataset
        df_relative.to_csv(new_file, index=False, encoding='windows-1253')
        
        print(f"\nΤΕΛΟΣ! Ο κόπος σου σώθηκε.")
        print(f"Το νέο αρχείο ονομάζεται: '{new_file}'")
        
    except Exception as e:
        print(f"Προέκυψε σφάλμα: {e}")

print("="*50 + "\n")