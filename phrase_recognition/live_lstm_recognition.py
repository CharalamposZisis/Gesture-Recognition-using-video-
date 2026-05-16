
import os
# ΑΥΤΟ ΕΙΝΑΙ ΤΟ ΜΥΣΤΙΚΟ ΚΟΛΠΟ! Πρέπει να μπει πριν τα υπόλοιπα imports
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

print("\n" + "="*60)
print("  ΦΟΡΤΩΣΗ ΕΞΥΠΝΟΥ ΣΥΣΤΗΜΑΤΟΣ (LIVE LSTM) ...")
print("="*60)

# 1. Φόρτωση του εκπαιδευμένου "εγκεφάλου"
try:
    model = load_model('best_lstm_model.h5')
    print("[*] Το μοντέλο φορτώθηκε επιτυχώς!")
except Exception as e:
    print(f"[!] Σφάλμα φόρτωσης μοντέλου: {e}")
    exit()

# 2. Ρυθμίσεις και Λεξικό Φράσεων (Πρέπει να ταιριάζει με τον Collector!)
MAX_SEQUENCE_LENGTH = 70
sequence = [] # Εδώ θα κρατάμε τα τελευταία 70 καρέ
current_phrase = ""
confidence_score = 0.0

# Προσοχή: Στην εκπαίδευση κάναμε τα labels από 1-5 σε 0-4.
# Οπότε το index 0 είναι το '1', το index 1 το '2' κ.ο.κ.
PHRASES = {
    0: 'Kalimera / Geia',
    1: 'Kalo vradu',
    2: 'Eisai kala? / Ti kaneis?',
    3: 'Kalos irthes',
    4: 'Pos se lene?'
}

# 3. Ξεκινάμε το MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def main():
    global sequence, current_phrase, confidence_score
    cap = cv2.VideoCapture(0)
    
    print("\nΗ κάμερα άνοιξε! Κάνε τις φράσεις σου.")
    print("Πάτα 'q' στο παράθυρο της κάμερας για έξοδο.\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        # Μαύρη μπάρα για το γραφικό UI
        cv2.rectangle(frame, (0, 0), (w, 85), (20, 20, 20), -1)

        # Πίνακας με 84 μηδενικά για το τρέχον καρέ (ακριβώς όπως στον collector)
        frame_data = np.zeros(84)

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                # Ζωγραφίζουμε τα χέρια
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Εξαγωγή συντεταγμένων
                hand_coords = []
                for lm in hand_landmarks.landmark:
                    hand_coords.extend([lm.x, lm.y])
                
                start_idx = i * 42
                end_idx = start_idx + 42
                frame_data[start_idx:end_idx] = hand_coords
        
        # Προσθέτουμε το καρέ στη μνήμη μας (sequence)
        sequence.append(frame_data.tolist())
        
        # Αν έχουμε μαζέψει πάνω από 70 καρέ, πετάμε το πιο παλιό (Κυλιόμενο Παράθυρο)
        if len(sequence) > MAX_SEQUENCE_LENGTH:
            sequence = sequence[-MAX_SEQUENCE_LENGTH:]

        # Μόλις το παράθυρο γεμίσει (φτάσει τα 70), ξεκινάμε τις προβλέψεις!
        if len(sequence) == MAX_SEQUENCE_LENGTH:
            # Το μετατρέπουμε στην 3D μορφή που θέλει το LSTM (1 βίντεο, 70 καρέ, 84 σημεία)
            input_data = np.array(sequence).reshape(1, 70, 84)
            
            # Το AI κάνει την πρόβλεψή του
            predictions = model.predict(input_data, verbose=0)[0]
            
            # Βρίσκουμε ποια φράση έχει το μεγαλύτερο ποσοστό
            best_match_index = np.argmax(predictions)
            confidence_score = predictions[best_match_index] * 100
            
            # Αν είναι πάνω από 80% σίγουρο, το τυπώνουμε. Αλλιώς ίσως είναι τυχαία κίνηση.
            if confidence_score > 80.0:
                current_phrase = PHRASES[best_match_index]
            else:
                current_phrase = "..." # Δεν είναι σίγουρο

        # --- ΕΜΦΑΝΙΣΗ ΣΤΗΝ ΟΘΟΝΗ (HUD) ---
        # Χρώμα ποσοστού ανάλογα με τη σιγουριά
        if confidence_score > 85:
            conf_color = (0, 255, 0) # Πράσινο
        elif confidence_score > 60:
            conf_color = (0, 255, 255) # Κίτρινο
        else:
            conf_color = (0, 0, 255) # Κόκκινο

        cv2.putText(frame, f"AI VISION | Phrase: {current_phrase}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    
        cv2.putText(frame, f"Confidence: {confidence_score:.1f}%", (20, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)

        # Μπάρα φόρτωσης του κυλιόμενου παραθύρου
        buffer_text = f"Buffer: {len(sequence)}/{MAX_SEQUENCE_LENGTH}"
        cv2.putText(frame, buffer_text, (w - 200, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        cv2.imshow("Live Phrase Recognition (LSTM)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()