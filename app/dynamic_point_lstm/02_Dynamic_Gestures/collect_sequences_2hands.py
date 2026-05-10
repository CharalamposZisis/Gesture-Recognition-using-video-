import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# ============================================================
# ΡΥΘΜΙΣΕΙΣ (2 Χέρια)
# ============================================================
csv_file = 'two_hands_sequences.csv'
sequence_length = 30 # Πόσα καρέ αποτελούν μια κίνηση
current_sequence = []

# Προετοιμασία CSV
if not os.path.isfile(csv_file):
    header = ['label']
    for i in range(sequence_length * 84):
        header.append(f'point_{i}')
        
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

# Ρυθμίζουμε το MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def main():
    global current_sequence
    cap = cv2.VideoCapture(0)
    recording = False
    label = ""
    count = 0

    print("\n" + "="*50)
    print("ΣΥΛΛΟΓΗ ΚΙΝΗΣΕΩΝ ΜΕ 2 ΧΕΡΙΑ (Visual Debug Enabled)")
    print("="*50)
    print("Πάτα '1', '2' ή '3' για να ξεκινήσεις καταγραφή.")
    print("Πάτα 'S' για παύση. Πάτα 'Q' για έξοδο.")
    print("="*50 + "\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        # Καθρεφτισμός της εικόνας για να είναι πιο φυσική η κίνηση
        frame = cv2.flip(frame, 1)

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        key = cv2.waitKey(1) & 0xFF

        # --- ΖΩΓΡΑΦΙΚΗ ΣΗΜΕΙΩΝ (Πάντα Ενεργή) ---
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        # ---------------------------------------

        # Επιλογή κίνησης
        if key in [ord('1'), ord('2'), ord('3')]:
            label = chr(key)
            recording = True
            current_sequence = []
            print(f"--- Καταγραφή κίνησης: {label} ---")

        if recording:
            # Προετοιμασία πίνακα με μηδενικά (padding)
            frame_data = np.zeros(84)
            
            if results.multi_hand_landmarks:
                # Παίρνουμε μέχρι 2 χέρια
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                    hand_coords = []
                    for lm in hand_landmarks.landmark:
                        hand_coords.extend([lm.x, lm.y])
                    
                    start_idx = i * 42
                    end_idx = start_idx + 42
                    frame_data[start_idx:end_idx] = hand_coords
            
            current_sequence.append(frame_data.tolist())
            
            cv2.putText(frame, f"REC: {label} ({len(current_sequence)}/{sequence_length})", 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            if len(current_sequence) == sequence_length:
                flat_data = [label] + [item for sublist in current_sequence for item in sublist]
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(flat_data)
                count += 1
                print(f"Αποθηκεύτηκε η κίνηση {label}! (Σύνολο: {count})")
                current_sequence = []

        if key == ord('s'): 
            recording = False
            current_sequence = []
            print("--- Παύση ---")
        if key == ord('q'): 
            break

        cv2.imshow("Two Hands Sequence Collector", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()