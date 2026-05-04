import cv2 # OpenCV: A library for computer vision tasks, used here for video capture and image processing.
import mediapipe as mp # mediapipe: Provides pre-trained models for hand detection.
import time # Used to measure time between frames to calculate FPS (frames per second)


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5, results=None):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands, # Set the maximum number of hands to detect in the image. This limits the number of hands that the model will attempt to detect and track.
            min_detection_confidence=self.detectionCon, # Set the minimum confidence threshold for hand detection. This value determines how confident the model must be in its detection before it considers it valid. A higher value means that only detections with high confidence will be accepted, while a lower value may allow for more detections but with a higher chance of false positives.
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        
        
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert the image from BGR to RGB color space.
        self.results = self.hands.process(imgRGB) # Process the RGB image to detect hands and extract landmarks.
        # print(self.results.multi_hand_landmarks) # Print the detected hand landmarks to the console.

        if self.results.multi_hand_landmarks: # If hand landmarks are detected, iterate through each detected hand.
            for handLms in self.results.multi_hand_landmarks: # For each detected hand, iterate through its landmarks.
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) # Draw the landmarks and connections on the hands. 

        return img            

# function to find the position of the hand landmarks and return a list of their coordinates
    def findPosition(self, img, handNo=0, draw=True):
        # Initialize an empty list to store the coordinates of the hand landmarks.
        lmList = []
        if self.results.multi_hand_landmarks: # If hand landmarks are detected, iterate through each detected hand (self. results we are getting it from the findHands method).
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.multi_hand_landmarks[handNo], self.mpHands.HAND_CONNECTIONS) # Draw the landmarks and connections on the image.
            myHand = self.results.multi_hand_landmarks[handNo] # Get the landmarks for the specified hand (handNo) from the results.
            for id, lm in enumerate(myHand.landmark): # For each landmark, calculate its pixel coordinates and print them.
                h, w, c = img.shape # Get the height, width, and number of channels of the image.
                cx, cy = int(lm.x * w), int(lm.y * h) # Calculate the pixel coordinates of the landmark by multiplying its normalized coordinates by the image dimensions.
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()  # If camera works capture an image
        if not success:
            print("Failed to capture image")
            break
        else:
            img = detector.findHands(img) # Call the findHands method to detect hands and draw landmarks on the image. If I set draw to False, it will not draw the landmarks on the image.
            cTime = time.time() # Get the current time to calculate FPS.
            fps = 1 / (cTime - pTime) # Calculate the frames per second (FPS) by taking the inverse of the time difference between the current and previous frames.
            pTime = cTime # Update the previous time to the current time for the next iteration.
            # Display the FPS on the image using OpenCV's putText function.
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv2.imshow("Image", img)
            cv2.waitKey(1)
            findPosition = detector.findPosition(img) # Call the findPosition method to get the coordinates of the hand landmarks and draw circles on them.
            if len(findPosition) != 0: # If the list of landmark coordinates is not empty, print the coordinates of the first landmark (id 0) to the console.
                print(findPosition[4]) # Print the coordinates of the first landmark (id 0) to the console.
            
            
if __name__ == "__main__":
    main()