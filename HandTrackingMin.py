import cv2
import mediapipe as mp # mediapipe: Provides pre-trained models for hand detection.
import time

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands # mp.solutions.hands: Accesses MediaPipe’s hand solutions.
hands = mp_hands.Hands() # hands = mp_hands.Hands(): Initializes the hand detection model.
mp_draw = mp.solutions.drawing_utils # mp.solutions.drawing_utils: Provides utilities for drawing landmarks on images.

pTime = 0
cTime = 0

while True:
    success, img = cap.read()  # If camera works capture an image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert the image from BGR to RGB color space.
    results = hands.process(imgRGB) # Process the RGB image to detect hands and extract
    print(results.multi_hand_landmarks) # Print the detected hand landmarks to the console.
    
    if results.multi_hand_landmarks: # If hand landmarks are detected, iterate through each detected hand.
        for handLms in results.multi_hand_landmarks: # For each detected hand, iterate through its landmarks.
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS) # Draw the landmarks and connections on the image.
            for id, lm in enumerate(handLms.landmark): # For each landmark, calculate its pixel coordinates and print them.
                h, w, c = img.shape # Get the height, width, and number of channels of the image.
                cx, cy = int(lm.x * w), int(lm.y * h) # Calculate the pixel coordinates of the landmark by multiplying its normalized coordinates by the image dimensions.
                print(id, cx, cy)
    
    cTime = time.time() # Get the current time to calculate FPS.
    fps = 1 / (cTime - pTime) # Calculate the frames per second (FPS) by taking the inverse of the time difference between the current and previous frames.
    pTime = cTime # Update the previous time to the current time for the next iteration.
    
    # Display the FPS on the image using OpenCV's putText function.
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)