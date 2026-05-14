import pandas as pd
import os

print("\n" + "="*50)
print("  ΕΡΓΑΛΕΙΟ ΔΙΑΓΡΑΦΗΣ ΤΕΛΕΥΤΑΙΑΣ ΕΓΓΡΑΦΗΣ")
print("="*50)

# Έξυπνος έλεγχος για να βρει το CSV
csv_file = 'phrase_recognition/sign_language_phrases.csv'
if not os.path.isfile(csv_file):
    csv_file = 'sign_language_phrases.csv'

if not os.path.isfile(csv_file):
    print("Σφάλμα: Δεν βρέθηκε το αρχείο CSV. Μήπως είναι άδειο;")
else:
    try:
        # Διαβάζουμε το CSV
        df = pd.read_csv(csv_file)
        
        if len(df) == 0:
            print("Το αρχείο είναι ήδη άδειο! Δεν υπάρχει κάτι να διαγραφεί.")
        else:
            # Βλέπουμε ποιο ήταν το τελευταίο label που γράφτηκε
            last_label = df.iloc[-1]['label']
            
            # ΚΟΒΟΥΜΕ την τελευταία γραμμή (κρατάμε όλες εκτός από την τελευταία)
            df = df.iloc[:-1]
            
            # Το αποθηκεύουμε ξανά (χωρίς το index)
            df.to_csv(csv_file, index=False)
            
            print(f"ΕΠΙΤΥΧΙΑ! Η τελευταία σου προσπάθεια (Φράση: {last_label}) ΔΙΑΓΡΑΦΗΚΕ.")
            print(f"Νέο σύνολο εγγραφών στο αρχείο: {len(df)}")
            
    except Exception as e:
        print(f"Προέκυψε σφάλμα: {e}")
        print("Σιγουρέψου ότι το αρχείο ΔΕΝ είναι ανοιχτό σε κάποιο άλλο πρόγραμμα (π.χ. Excel).")

print("="*50 + "\n")