import pandas as pd
import os

print("\n" + "="*65)
print("  ΕΡΓΑΛΕΙΟ ΕΝΩΣΗΣ ΠΟΛΛΑΠΛΩΝ CSV (SMART MASTER MERGER)")
print("="*65)

# --- Βάλε εδώ τα ονόματα των 3 (ή παραπάνω) αρχείων σου ---
files_to_merge = [
    'sign_language_phrases_tasos.csv', 
    'sign_language_phrases_chris.csv', 
    'sign_language_phrases_fixed.csv'
]

# Το όνομα του τελικού, τεράστιου αρχείου που θα προκύψει
output_file = 'master_phrases_dataset.csv'

def read_csv_smart(filename):
    # Δοκιμάζει αυτόματα όλες τις κωδικοποιήσεις για να μην σκάσει πουθενά
    encodings_to_try = ['utf-8-sig', 'utf-8', 'windows-1253', 'latin1']
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(filename, encoding=enc)
            print(f"   -> Επιτυχία: Διαβάστηκε με {enc} | Γραμμές: {len(df)}")
            return df
        except Exception:
            continue
    raise ValueError(f"Το αρχείο δεν μπόρεσε να διαβαστεί με καμία γνωστή κωδικοποίηση.")

dataframes = []
total_files = len(files_to_merge)

# Διαβάζουμε ένα-ένα τα αρχεία της λίστας
for i, file in enumerate(files_to_merge, 1):
    if not os.path.isfile(file):
        print(f"[{i}/{total_files}] Προσοχή: Το '{file}' δεν βρέθηκε. Θα το προσπεράσω.")
        continue
    
    print(f"[{i}/{total_files}] Φόρτωση: {file}")
    try:
        df = read_csv_smart(file)
        dataframes.append(df)
    except Exception as e:
        print(f"   [Σφάλμα] {e}")

# Αν βρήκε έστω και ένα σωστό αρχείο, τα ενώνει όλα μαζί
if len(dataframes) > 0:
    print("\n[*] Ξεκινάει η συγκόλληση όλων των αρχείων...")
    # Το ignore_index=True φροντίζει να φτιάξει σωστή αρίθμηση από την αρχή
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # Το αποθηκεύουμε με utf-8-sig για να παίζει τέλεια παντού (και στο Excel)
    merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("\nΕΠΙΤΥΧΙΑ! Το Master Dataset δημιουργήθηκε.")
    print(f" -> ΤΕΛΙΚΟ ΣΥΝΟΛΟ ΒΙΝΤΕΟ (Γραμμές) στο '{output_file}': {len(merged_df)}")
else:
    print("\n[!] Αποτυχία: Δεν βρέθηκε κανένα έγκυρο αρχείο για ένωση. Έλεγξε τα ονόματα!")

print("="*65 + "\n")