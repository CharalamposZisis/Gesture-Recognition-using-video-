import cv2
import mediapipe as mp
import csv
import os
import time

# ============================================================
# ΡΥΘΜΙΣΕΙΣ ΑΡΧΕΙΟΥ CSV
# ============================================================
csv_file = 'greek_hand_dataset.csv' # Αλλάξαμε και το όνομα του αρχείου!

if not os.path.isfile(csv_file):
    columns = ['label']
    for i in range(21):
        columns.extend([f'x{i}', f'y{i}'])
        
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)

# ============================================================
# SETUP MEDIAPIPE
# ============================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def main():
    cap = cv2.VideoCapture(0)
    
    print("\n" + "="*50)
    print("ΣΥΛΛΟΓΗ ΔΕΔΟΜΕΝΩΝ - ΕΛΛΗΝΙΚΗ ΝΟΗΜΑΤΙΚΗ (Α, Β, Γ)")
    print("="*50)
    print("1. Πάτα 'A', 'B' ή 'G' (για το Γάμμα) για ΕΝΑΡΞΗ καταγραφής.")
    print("2. Κούνα το χέρι σου αργά στην κάμερα.")
    print("3. Πάτα 'S' για ΠΑΥΣΗ καταγραφής.")
    print("4. Πάτα 'Q' για ΕΞΟΔΟ.")
    print("="*50 + "\n")

    counters = {'A': 0, 'B': 0, 'G': 0}
    
    recording = False
    current_label = ""
    last_save_time = 0
    save_interval = 0.1  # Αποθήκευση 10 φορές το δευτερόλεπτο

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        key = cv2.waitKey(1) & 0xFF

        # Διακόπτης για τα γράμματα A, B και G (Γάμμα)
        if key in [ord('a'), ord('b'), ord('g')]:
            current_label = chr(key).upper()
            recording = True
            print(f"--- ΓΡΑΦΟΥΜΕ ΤΟ ΓΡΑΜΜΑ: '{current_label}' ---")
        elif key == ord('s'):
            recording = False
            current_label = ""
            print("--- Η ΚΑΤΑΓΡΑΦΗ ΣΤΑΜΑΤΗΣΕ ---")
        elif key == ord('q'):
            break

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            current_time = time.time()
            if recording and (current_time - last_save_time > save_interval):
                row = [current_label]
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y])
                
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                
                counters[current_label] += 1
                last_save_time = current_time

        # Ενδείξεις στην οθόνη
        if recording:
            cv2.putText(frame, f"REC: {current_label}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        cv2.putText(frame, f"A (Alpha): {counters['A']}  B (Vita): {counters['B']}  G (Gamma): {counters['G']}", 
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Greek Sign Language - Data Collection", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()