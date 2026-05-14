import cv2
import mediapipe as mp
import csv
import os
import time
import numpy as np

# ============================================================
# ΡΥΘΜΙΣΕΙΣ (2 Χέρια)
# ============================================================
csv_file = 'phrase_recognition/sign_language_phrases.csv'
countdown_seconds = 3  # Δευτερόλεπτα καθυστέρησης πριν ξεκινήσει η καταγραφή

# ΛΕΞΙΚΟ ΦΡΑΣΕΩΝ
PHRASES = {
    '1': 'Καλημέρα/Καλησπέρα/Γειά',
    '2': 'Καλό βράδυ',
    '3': 'Είσαι καλά/Τι κάνεις;',
    '4': 'Καλώς ήρθες',
    '5': 'Πώς σε λένε;',
}

# ΜΗΚΟΣ ΑΚΟΛΟΥΘΙΑΣ ΑΝΑ ΦΡΑΣΗ (πόσα frames καταγράφουμε)
SEQUENCE_LENGTHS = {
    '1': 50,
    '2': 50,
    '3': 70,   # μεγαλύτερη φράση -> περισσότερα frames
    '4': 70,
    '5': 60,
}

# Το μέγιστο μήκος καθορίζει το μέγεθος του CSV (padding για μικρότερα)
MAX_SEQUENCE_LENGTH = max(SEQUENCE_LENGTHS.values())
current_sequence = []

# Προετοιμασία CSV (μέγεθος βάσει του μέγιστου sequence)
if not os.path.isfile(csv_file):
    header = ['label']
    for i in range(MAX_SEQUENCE_LENGTH * 84):
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
    counting_down = False
    countdown_start_time = 0.0
    label = ""
    count = 0
    phrase_counts = {}

    print("\n" + "=" * 50)
    print("SIGN LANGUAGE PHRASE COLLECTOR")
    print("=" * 50)
    print("\nAvailable phrases:")
    for key, phrase in PHRASES.items():
        print(f"  {key}: {phrase} ({SEQUENCE_LENGTHS[key]} frames)")
    print(f"\nMax sequence length (CSV size): {MAX_SEQUENCE_LENGTH}")
    print("\nCommands:")
    print("  Press 1-5 to start recording (with countdown)")
    print("  Press S to stop / cancel countdown")
    print("  Press C to clear")
    print("  Press Q to exit")
    print("=" * 50 + "\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        key = cv2.waitKey(1) & 0xFF

        # --- ΖΩΓΡΑΦΙΚΗ ΣΗΜΕΙΩΝ ---
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # --- ΕΠΙΛΟΓΗ ΦΡΑΣΗΣ -> COUNTDOWN ---
        if chr(key) in PHRASES and not recording and not counting_down:
            label = chr(key)
            counting_down = True
            countdown_start_time = time.time()
            current_sequence = []
            print(f"--- Countdown started for: {label} - {PHRASES[label]} "
                  f"({SEQUENCE_LENGTHS[label]} frames) ---")

        # --- ΦΑΣΗ COUNTDOWN ---
        if counting_down:
            elapsed = time.time() - countdown_start_time
            remaining = countdown_seconds - elapsed

            if remaining <= 0:
                counting_down = False
                recording = True
                current_sequence = []
                print(f"--- Recording: {label} - {PHRASES[label]} ---")
            else:
                cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
                cv2.putText(frame, f"GET READY: {PHRASES[label]}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                count_num = int(remaining) + 1
                text = str(count_num)
                font_scale = 6
                thickness = 12
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                              font_scale, thickness)
                cx = (w - tw) // 2
                cy = (h + th) // 2
                cv2.putText(frame, text, (cx + 4, cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 4)
                cv2.putText(frame, text, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

                cv2.putText(frame, "Press S to cancel",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        # --- ΦΑΣΗ RECORDING ---
        elif recording:
            target_length = SEQUENCE_LENGTHS[label]  # μήκος για τη συγκεκριμένη φράση
            frame_data = np.zeros(84)

            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                    hand_coords = []
                    for lm in hand_landmarks.landmark:
                        hand_coords.extend([lm.x, lm.y])

                    start_idx = i * 42
                    end_idx = start_idx + 42
                    frame_data[start_idx:end_idx] = hand_coords

            current_sequence.append(frame_data.tolist())

            cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
            cv2.putText(frame, f"REC: {PHRASES[label]}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(frame, f"Frames: {len(current_sequence)}/{target_length}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if len(current_sequence) == target_length:
                # PADDING: γεμίζουμε με μηδενικά μέχρι το MAX_SEQUENCE_LENGTH
                padded_sequence = list(current_sequence)
                while len(padded_sequence) < MAX_SEQUENCE_LENGTH:
                    padded_sequence.append([0.0] * 84)

                flat_data = [label] + [item for sublist in padded_sequence for item in sublist]
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(flat_data)
                phrase_counts[label] = phrase_counts.get(label, 0) + 1
                count += 1
                print(f"SAVED: {label} - {PHRASES[label]} "
                      f"[{target_length} real + {MAX_SEQUENCE_LENGTH - target_length} padding] "
                      f"(Total: {count})")
                current_sequence = []
                recording = False
                label = ""
        else:
            cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
            cv2.putText(frame, "READY - Press 1-5 to record",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

        # Εμφάνιση λίστας φράσεων
        y_offset = 20
        for k in sorted(PHRASES.keys()):
            cnt = phrase_counts.get(k, 0)
            color = (0, 255, 0) if cnt > 0 else (100, 100, 100)
            text = f"{k}: {PHRASES[k]} [{SEQUENCE_LENGTHS[k]}f] ({cnt})"
            cv2.putText(frame, text, (w - 360, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            y_offset += 25

        cv2.putText(frame, "S: Stop | C: Clear | Q: Quit", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        cv2.imshow("Sign Language Phrase Collector", frame)

        if key == ord('s') or key == ord('S'):
            recording = False
            counting_down = False
            current_sequence = []
            label = ""
            print("--- Stopped ---")

        if key == ord('c') or key == ord('C'):
            current_sequence = []
            print("--- Cleared ---")

        if key == ord('q') or key == ord('Q'):
            print("--- Exiting ---")
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 50)
    print("Session Complete!")
    print(f"Total phrases recorded: {count}")
    print("\nBreakdown:")
    for k in sorted(PHRASES.keys()):
        cnt = phrase_counts.get(k, 0)
        print(f"  {k}: {PHRASES[k]:30s} [{SEQUENCE_LENGTHS[k]}f] - {cnt}")
    print(f"\nData saved to: {csv_file}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()