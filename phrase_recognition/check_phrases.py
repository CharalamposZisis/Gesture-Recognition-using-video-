import pandas as pd
import os

print("\n" + "="*60)
print("  ΣΤΑΤΙΣΤΙΚΑ ΔΕΔΟΜΕΝΩΝ (ΑΝΑ ΦΡΑΣΗ)")
print("="*60)

# Το λεξικό με τις φράσεις (όπως το έχεις στον collector)
PHRASES = {
    '1': 'Καλημέρα/Καλησπέρα/Γειά',
    '2': 'Καλό βράδυ',
    '3': 'Είσαι καλά/Τι κάνεις;',
    '4': 'Καλώς ήρθες',
    '5': 'Πώς σε λένε;'
}

# Έξυπνος έλεγχος για να βρει το CSV όπου κι αν έχεις τρέξει το script
csv_file = 'phrase_recognition/sign_language_phrases.csv'
if not os.path.isfile(csv_file):
    csv_file = 'sign_language_phrases.csv'

if not os.path.isfile(csv_file):
    print(f"Σφάλμα: Δεν βρέθηκε το αρχείο CSV.")
    print("Μήπως δεν έχεις γράψει ακόμα καμία φράση;")
else:
    try:
        # Διαβάζουμε το CSV
        df = pd.read_csv(csv_file)
        
        # Επειδή το pandas μπορεί να διαβάσει το '1' ως αριθμό 1, το μετατρέπουμε σε κείμενο (string) για σιγουριά
        df['label'] = df['label'].astype(str)
        
        # Μετράμε πόσες φορές εμφανίζεται η κάθε φράση
        counts = df['label'].value_counts()
        
        total_samples = len(df)
        print(f"Συνολικές εγγραφές (βίντεο) στο αρχείο: {total_samples}\n")
        
        # Τυπώνουμε τα αποτελέσματα με τη σειρά του λεξικού (1 έως 5)
        for key in sorted(PHRASES.keys()):
            phrase_text = PHRASES[key]
            # Παίρνουμε τον αριθμό εγγραφών (αν δεν υπάρχει, βάζουμε 0)
            count = counts.get(key, 0) 
            
            print(f" [{key}] {phrase_text:30s} : \t {count} εγγραφές")
            
        print("-" * 60)
        print(f"Διαφορετικές φράσεις που βρέθηκαν: {len(counts)}")
        
    except Exception as e:
        print(f"Προέκυψε ένα σφάλμα κατά την ανάγνωση: {e}")

print("="*60 + "\n")