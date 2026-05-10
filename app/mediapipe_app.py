import cv2
import mediapipe as mp
import time

# ============================================================
# SETUP MEDIAPIPE
# ============================================================
mp_hands = mp.solutions.hands
# Βάζουμε 1 χέρι προς το παρόν για να είναι πιο εύκολοι οι υπολογισμοί
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Οι αριθμοί των σημείων που βρίσκονται στις άκρες των 4 δαχτύλων 
# (Δείκτης=8, Μέσος=12, Παράμεσος=16, Μικρό=20)
tipIds = [8, 12, 16, 20]

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    
    print("\n" + "="*50)
    print("ΑΝΑΓΝΩΡΙΣΗ ΝΟΗΜΑΤΙΚΗΣ - LEVEL 1")
    print("="*50 + "\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        letter = "" # Εδώ θα αποθηκεύουμε το γράμμα που βρίσκει

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Παίρνουμε τις (x, y) συντεταγμένες όλων των 21 σημείων
                lmList = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    
                if len(lmList) == 21:
                    fingers = []
                    
                    # 1. ΕΛΕΓΧΟΣ ΑΝΤΙΧΕΙΡΑ (Συγκρίνουμε δεξιά-αριστερά, άξονας X)
                    # (Ισχύει κυρίως για το δεξί χέρι)
                    if lmList[4][1] > lmList[3][1]:
                        fingers.append(1) # Ανοιχτός
                    else:
                        fingers.append(0) # Κλειστός

                    # 2. ΕΛΕΓΧΟΣ ΓΙΑ ΤΑ ΑΛΛΑ 4 ΔΑΧΤΥΛΑ (Συγκρίνουμε πάνω-κάτω, άξονας Y)
                    for id in range(0, 4):
                        # Αν το y της άκρης είναι μικρότερο (πιο ψηλά στην οθόνη) από την κάτω κλείδωση
                        if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                            fingers.append(1) # Ανοιχτό
                        else:
                            fingers.append(0) # Κλειστό

                    # Τώρα έχουμε μια λίστα με 5 αριθμούς (1=ανοιχτό, 0=κλειστό)
                    # Παράδειγμα: [Αντίχειρας, Δείκτης, Μέσος, Παράμεσος, Μικρό]
                    
                    # 3. ΚΑΝΟΝΕΣ ΓΙΑ ΤΑ ΓΡΑΜΜΑΤΑ (Αμερικανική Νοηματική - ASL)
                    if fingers == [0, 0, 0, 0, 0]:
                        letter = "A" # Όλα κλειστά (μπουνιά)
                    elif fingers == [0, 1, 1, 1, 1]:
                        letter = "B" # Όλα ανοιχτά (εκτός αντίχειρα)
                    elif fingers == [0, 1, 0, 0, 0]:
                        letter = "D" # Μόνο ο δείκτης πάνω
                    elif fingers == [0, 1, 1, 0, 0]:
                        letter = "V" # Δείκτης και Μέσος πάνω (Σήμα νίκης)
                    elif fingers == [1, 0, 0, 0, 1]:
                        letter = "Y" # Αντίχειρας και Μικρό (Σήμα "τηλέφωνο" / Shaka)

        # ============================================================
        # ΕΜΦΑΝΙΣΗ ΣΤΗΝ ΟΘΟΝΗ
        # ============================================================
        
        # Ζωγραφίζουμε ένα πράσινο κουτί πάνω αριστερά
        cv2.rectangle(frame, (20, 20), (150, 150), (0, 255, 0), cv2.FILLED)
        
        # Γράφουμε το γράμμα μέσα στο κουτί
        cv2.putText(frame, letter, (45, 110), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)

        # Υπολογισμός FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(frame, f"FPS: {int(fps)}", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Sign Language Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()