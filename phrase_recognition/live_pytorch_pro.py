import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque

print("\n" + "="*65)
print("  ΕΚΚΙΝΗΣΗ LIVE ΣΥΣΤΗΜΑΤΟΣ ΑΝΑΓΝΩΡΙΣΗΣ (PyTorch PRO)")
print("="*65)

# --- 1. ΟΡΙΣΜΟΣ ΜΟΝΤΕΛΟΥ ---
class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GestureLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Αφαιρέσαμε το dropout από το live, δεν χρειάζεται στο inference
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# --- 2. ΡΥΘΜΙΣΕΙΣ & ΦΟΡΤΩΣΗ ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Σύστημα Εκτέλεσης: {device.type.upper()}")

input_size = 84
hidden_size = 128
num_layers = 2
num_classes = 5
sequence_length = 70

# Τα Greeklish μας (το OpenCV δεν αγαπάει τα Ελληνικά χωρίς ειδικά fonts)
actions = [
    'Kalimera / Geia', 
    'Kalo vradu', 
    'Eisai kala? / Ti kaneis?', 
    'Kalos irthes', 
    'Pos se lene?'
]

try:
    model = GestureLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    # Προσοχή: Βάλε το σωστό όνομα του αρχείου που έκανες save στην εκπαίδευση
    model.load_state_dict(torch.load('best_pytorch_phrase_model.pth', map_location=device))
    model.eval() # Πολύ σημαντικό: Βάζει το μοντέλο σε "Λειτουργία Εξέτασης"
    print("[*] Ο 'εγκέφαλος' φορτώθηκε επιτυχώς!")
except Exception as e:
    print(f"[!] Σφάλμα φόρτωσης: {e}")
    exit()

# --- 3. SETUP ΚΑΜΕΡΑΣ & MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Το deque κρατάει ΑΥΤΟΜΑΤΑ μόνο τα τελευταία 70 καρέ! Πετάει τα παλιά μόνο του.
sequence_buffer = deque(maxlen=sequence_length)

def main():
    global sequence_buffer
    cap = cv2.VideoCapture(0)
    
    # Ρύθμιση ανάλυσης κάμερας (προαιρετικό, αλλά κάνει το UI πιο ευρύχωρο)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n[>] Η κάμερα άνοιξε! Κάνε τις φράσεις σου.")
    print("[>] Πάτα 'Q' για έξοδο.\n")

    current_action = "Waiting for buffer..."
    confidence_score = 0.0

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1) # Καθρέφτης για να μην μπερδεύεσαι
        h, w, _ = frame.shape
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        frame_data = np.zeros(84)
        
        # --- ΕΝΤΟΠΙΣΜΟΣ ΧΕΡΙΩΝ ---
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                # Ζωγραφική του σκελετού με λίγο πιο ωραία χρώματα
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
                
                hand_coords = []
                for lm in hand_landmarks.landmark:
                    hand_coords.extend([lm.x, lm.y])
                
                start_idx = i * 42
                frame_data[start_idx:start_idx+42] = hand_coords

        sequence_buffer.append(frame_data)

        # --- ΠΡΟΒΛΕΨΗ AI (Μόνο αν γεμίσει το buffer) ---
        if len(sequence_buffer) == sequence_length:
            input_data = torch.tensor([list(sequence_buffer)], dtype=torch.float32).to(device)
            
            with torch.no_grad():
                output = model(input_data)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                conf_val = confidence.item() * 100
                
                # ΟΡΙΑ ΣΙΓΟΥΡΙΑΣ (Thresholds)
                if conf_val > 85.0:
                    current_action = actions[predicted.item()]
                    confidence_score = conf_val
                else:
                    current_action = "Listening..."
                    confidence_score = conf_val

        # ==========================================
        #   ΕΠΑΓΓΕΛΜΑΤΙΚΟ UX / UI (Heads-Up Display)
        # ==========================================
        
        # 1. Μαύρη ημιδιαφανής μπάρα πάνω (για να ξεχωρίζουν τα γράμματα)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # 2. Χρωματισμός ποσοστού (Traffic Light logic)
        if confidence_score > 85.0:
            color = (0, 255, 0)      # Πράσινο (Σίγουρο)
        elif confidence_score > 50.0:
            color = (0, 255, 255)    # Κίτρινο (Σκέφτεται)
        else:
            color = (150, 150, 150)  # Γκρι (Αδρανές/Θόρυβος)

        # 3. Εμφάνιση Κειμένου (Κύρια Φράση)
        if current_action not in ["Waiting for buffer...", "Listening..."]:
            cv2.putText(frame, f"Phrase: {current_action}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(frame, f"Confidence: {confidence_score:.1f}%", (20, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, current_action, (20, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

        # 4. Μπάρα Buffer (Κάτω δεξιά)
        buffer_percent = len(sequence_buffer) / sequence_length
        bar_width = 200
        bar_height = 15
        x_start = w - bar_width - 20
        y_start = 40
        
        # Περίγραμμα Μπάρας
        cv2.rectangle(frame, (x_start, y_start), (x_start + bar_width, y_start + bar_height), (255, 255, 255), 1)
        # Γέμισμα Μπάρας
        cv2.rectangle(frame, (x_start, y_start), (x_start + int(bar_width * buffer_percent), y_start + bar_height), (0, 200, 255), -1)
        # Κείμενο Μπάρας
        cv2.putText(frame, f"Buffer: {len(sequence_buffer)}/{sequence_length}", (x_start, y_start - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Live Sign Language Translation (PRO)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()