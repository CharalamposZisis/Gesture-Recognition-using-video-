import pandas as pd
import os

print("\n" + "="*60)
print("  ΕΡΓΑΛΕΙΟ ΕΝΩΣΗΣ CSV (ΜΕ ΑΥΤΟΜΑΤΗ ΑΝΑΓΝΩΡΙΣΗ ΚΩΔΙΚΟΠΟΙΗΣΗΣ)")
print("="*60)

file1 = 'greek_hand_dataset_relative_mpampis.csv'
file2 = 'greek_hand_dataset_relative_chris.csv'
output_file = 'final_merged_dataset_relative.csv'

def read_csv_smart(filename):
    # Λίστα με τις πιο γνωστές κωδικοποιήσεις. Το πρόγραμμα θα τις δοκιμάσει με τη σειρά.
    encodings_to_try = ['utf-8-sig', 'utf-8', 'windows-1253', 'latin1']
    
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(filename, encoding=enc)
            print(f"   -> Επιτυχία! Το αρχείο διαβάστηκε με κωδικοποίηση: {enc}")
            return df
        except Exception:
            continue # Αν σκάσει, δοκιμάζει την επόμενη κωδικοποίηση
            
    # Αν φτάσει εδώ, σημαίνει ότι απέτυχαν όλες!
    raise ValueError(f"Το αρχείο '{filename}' είναι είτε άδειο, είτε εντελώς κατεστραμμένο.")

if not os.path.isfile(file1) or not os.path.isfile(file2):
    print("Σφάλμα: Κάποιο από τα δύο αρχεία δεν βρέθηκε στον φάκελο.")
else:
    try:
        print(f"[1/3] Φόρτωση 1ου αρχείου: {file1}")
        df1 = read_csv_smart(file1)
        
        print(f"[2/3] Φόρτωση 2ου αρχείου: {file2}")
        df2 = read_csv_smart(file2)
        
        print("[3/3] Συγκόλληση και Αποθήκευση...")
        # Τα ενώνουμε (βάζουμε το ένα κάτω από το άλλο)
        merged_df = pd.concat([df1, df2], ignore_index=True)
        
        # Εξαγωγή με utf-8-sig για να παίζει τέλεια παντού
        merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print("\nΕΠΙΤΥΧΙΑ! Το τελικό αρχείο δημιουργήθηκε.")
        print(f" -> Γραμμές 1ου: {len(df1)}")
        print(f" -> Γραμμές 2ου: {len(df2)}")
        print(f" -> ΣΥΝΟΛΟ στο '{output_file}': {len(merged_df)}")
        
    except Exception as e:
        print(f"\nΣφάλμα: {e}")

print("="*60 + "\n")