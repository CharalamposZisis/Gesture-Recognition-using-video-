import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

# --- 1. ΟΡΙΣΜΟΣ ΜΟΝΤΕΛΟΥ ---
class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GestureLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# --- 2. ΦΟΡΤΩΣΗ ---
input_size = 84
hidden_size = 64
num_layers = 2
num_classes = 3

model = GestureLSTM(input_size, hidden_size, num_layers, num_classes)
model.load_state_dict(torch.load('gesture_lstm.pth'))
model.eval()

# --- 3. SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

sequence_buffer = []
sequence_length = 30
actions = ['WAVE', 'TIME OUT','IDLE']

def main():
    global sequence_buffer
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        frame_data = np.zeros(84)
        
        # --- ΕΔΩ ΣΧΕΔΙΑΖΟΥΜΕ ΤΙΣ ΚΟΥΚΙΔΕΣ ---
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                # Αυτή η γραμμή ζωγραφίζει τον σκελετό στο frame
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                hand_coords = []
                for lm in hand_landmarks.landmark:
                    hand_coords.extend([lm.x, lm.y])
                
                start_idx = i * 42
                frame_data[start_idx:start_idx+42] = hand_coords

        sequence_buffer.append(frame_data)
        sequence_buffer = sequence_buffer[-sequence_length:]

        current_action = "" # Ξεκινάμε με κενό
        confidence_text = ""

        if len(sequence_buffer) == sequence_length:
            input_data = torch.tensor([sequence_buffer], dtype=torch.float32)
            with torch.no_grad():
                output = model(input_data)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # --- ΕΔΩ ΕΙΝΑΙ ΤΟ ΦΙΛΤΡΟ ΣΙΓΟΥΡΙΑΣ ---
                # Αν η σιγουριά είναι κάτω από 0.9 (90%), δεν δείχνει τίποτα
                if confidence.item() > 0.9:
                    current_action = actions[predicted.item()]
                    confidence_text = f"{int(confidence.item() * 100)}%"

        # Εμφάνιση UI
        if current_action:
            cv2.rectangle(frame, (0,0), (320, 80), (0, 255, 0), -1)
            cv2.putText(frame, current_action, (15, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
            cv2.putText(frame, confidence_text, (15, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        else:
            # Αν δεν είναι σίγουρο, δείχνουμε μια διακριτική ένδειξη
            cv2.putText(frame, "Waiting for gesture...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Live Recognition - Master Edition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()