import cv2
import mediapipe as mp
import pandas as pd
import pickle

print("\n" + "="*50)
print("ΦΟΡΤΩΣΗ ΕΞΥΠΝΟΥ ΣΥΣΤΗΜΑΤΟΣ...")
print("="*50)

# 1. Φορτώνουμε τον "εγκέφαλο" (το εκπαιδευμένο μοντέλο)
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. Φτιάχνουμε τα ονόματα των στηλών (όπως ακριβώς ήταν στο Excel)
# για να καταλαβαίνει το μοντέλο τι του δίνουμε
columns = []
for i in range(21):
    columns.extend([f'x{i}', f'y{i}'])

# 3. Ξεκινάμε το MediaPipe για να βλέπει τον σκελετό
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def main():
    cap = cv2.VideoCapture(0)
    
    print("Η κάμερα άνοιξε! Κάνε τα γράμματα Α, Β ή Γ μπροστά της.")
    print("Πάτα 'q' στο παράθυρο της κάμερας για έξοδο.\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Ζωγραφίζουμε τον σκελετό
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Εξάγουμε τις 42 συντεταγμένες (x, y) από τα 21 σημεία
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y])
            
            # Τα βάζουμε στη μορφή που τα θέλει το μοντέλο
            X_live = pd.DataFrame([row], columns=columns)
            
            # Του ζητάμε να μαντέψει τι βλέπει!
            prediction = model.predict(X_live)[0]
            
            # Μια μικρή διόρθωση για την οθόνη (για να βλέπουμε τη λέξη Gamma)
            display_text = prediction
            if prediction == 'G':
                display_text = 'G (Gamma)'
            
            # Εμφανίζουμε την πρόβλεψη στην οθόνη με τεράστια πράσινα γράμματα
            cv2.putText(frame, f"Letter: {display_text}", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

        cv2.imshow("Live Sign Language AI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()