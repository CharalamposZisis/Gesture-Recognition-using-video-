import csv
import os

print("\n" + "="*55)
print("  ΕΠΙΧΕΙΡΗΣΗ 'ΔΙΑΣΩΣΗ' - ΕΠΙΔΙΟΡΘΩΣΗ FRANKENSTEIN CSV")
print("="*55)

input_file = 'sign_language_phrases.csv'
output_file = 'sign_language_phrases_fixed.csv'

if not os.path.isfile(input_file):
    print(f"Σφάλμα: Δεν βρέθηκε το αρχείο {input_file}")
else:
    fixed_count = 0
    ok_count = 0
    skipped_count = 0

    # Ανοίγουμε το χαλασμένο για διάβασμα και ένα νέο για γράψιμο
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_file, mode='w', newline='', encoding='utf-8-sig') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for i, row in enumerate(reader):
            # Η πρώτη γραμμή είναι οι τίτλοι των στηλών. Τους ξαναφτιάχνουμε σωστά!
            if i == 0 and row[0] == 'label':
                header = ['label'] + [f'point_{j}' for j in range(70 * 84)]
                writer.writerow(header)
                continue

            row_len = len(row)
            
            if row_len == 4201:
                # Είναι παλιό (50 frames). Κολλάμε 1680 μηδενικά (20 frames * 84 σημεία)
                row.extend(['0.0'] * 1680)
                writer.writerow(row)
                fixed_count += 1
            elif row_len == 5881:
                # Είναι ήδη σωστό (70 frames). Το γράφουμε όπως είναι.
                writer.writerow(row)
                ok_count += 1
            else:
                # Αν υπάρχει κάποια τελείως κατεστραμμένη γραμμή, την προσπερνάμε
                skipped_count += 1

    print(f"\nΗ επιχείρηση διάσωσης ολοκληρώθηκε με επιτυχία!")
    print(f" -> Αναβαθμίστηκαν (από 50 σε 70 frames): {fixed_count} βίντεο")
    print(f" -> Ήταν ήδη τέλεια (70 frames): {ok_count} βίντεο")
    if skipped_count > 0:
        print(f" -> Απορρίφθηκαν ως άκυρα: {skipped_count} γραμμές")
    print(f"\nΤο ΥΓΙΕΣ αρχείο σώθηκε ως: '{output_file}'")
    print("="*55 + "\n")