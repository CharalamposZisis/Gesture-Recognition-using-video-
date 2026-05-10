import pandas as pd
import os

print("\n" + "="*40)
print("  ΣΤΑΤΙΣΤΙΚΑ ΔΕΔΟΜΕΝΩΝ (ΑΝΑ ΓΡΑΜΜΑ)")
print("="*40)

csv_file = 'greek_hand_dataset.csv'

# Επιβάλλουμε τη σωστή σειρά του ελληνικού αλφαβήτου "με το ζόρι"
GREEK_ALPHABET = [
    'A', 'B', 'Γ', 'Δ', 'E', 'Z', 'H', 'Θ', 'I', 'K', 'Λ', 'M', 
    'N', 'Ξ', 'O', 'Π', 'P', 'Σ', 'T', 'Y', 'Φ', 'X', 'Ψ', 'Ω'
]

if not os.path.isfile(csv_file):
    print("Σφάλμα: Δεν βρέθηκε το αρχείο 'greek_hand_dataset.csv'.")
    print("Μήπως δεν έχεις ξεκινήσει ακόμα τη συλλογή;")
else:
    try:
        # Διαβάζουμε το αρχείο με τη σωστή κωδικοποίηση
        df = pd.read_csv(csv_file, encoding='windows-1253')
        
        # Μετράμε πόσες φορές εμφανίζεται το κάθε γράμμα
        counts = df['label'].value_counts()
        
        # Υπολογίζουμε το συνολικό πλήθος
        total_samples = len(df)
        
        print(f"Συνολικά δείγματα στο αρχείο: {total_samples}\n")
        
        # Τυπώνουμε τα αποτελέσματα βασισμένα στη ΔΙΚΗ ΜΑΣ λίστα (GREEK_ALPHABET)
        for letter in GREEK_ALPHABET:
            if letter in counts:
                print(f" Γράμμα '{letter}': \t {counts[letter]} δείγματα")
            
        print("-" * 40)
        print(f"Διαφορετικά γράμματα που βρέθηκαν: {len(counts)}")
        
    except Exception as e:
        print(f"Προέκυψε ένα σφάλμα κατά την ανάγνωση: {e}")

print("="*40 + "\n")