# SIGN LANGUAGE RECOGNITION APP - FIXED
# Χρησιμοποιεί YOLOv5 trained model (.pt)

import cv2
import torch
import numpy as np
import time

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
# ============================================================
# SETUP
# ============================================================

# Φόρτωσε το YOLOv5 model
MODEL_PATH = 'best.pt'
CONFIDENCE_THRESHOLD = 0.5

print("Loading YOLOv5 model...")
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
    model.conf = CONFIDENCE_THRESHOLD  # Set confidence threshold
    print(f"✓ Model loaded: {MODEL_PATH}")
    print(f"✓ Classes: {model.names}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("Make sure you have: pip install yolov5")
    exit()

# ============================================================
# FUNCTIONS
# ============================================================

def detect_sign_language(frame, model):
    """
    Κάνε detection με YOLOv5
    
    Returns:
        - results: YOLOv5 results
        - annotated_frame: frame με τα bounding boxes
    """
    # YOLOv5 detection
    results = model(frame)
    
    # Render - αυτό ζωγραφίζει τα boxes
    annotated_frame = results.render()[0]
    
    # ✅ ΣΗΜΑΝΤΙΚΟ: Κάνε το frame writable (όχι read-only)
    annotated_frame = annotated_frame.copy()
    
    return results, annotated_frame

def extract_predictions(results, conf_threshold=0.5):
    """
    Εξάγε τις προβλέψεις από τα YOLOv5 results
    
    Returns:
        - predictions: list of dicts with class, confidence, bbox
    """
    predictions = []
    
    try:
        # Παίρνου τα detected objects
        detections = results.xyxy[0]  # (x1, y1, x2, y2, confidence, class)
        
        if len(detections) > 0:
            for detection in detections:
                x1, y1, x2, y2, conf, cls_idx = detection
                
                # Έλεγχος confidence
                if conf >= conf_threshold:
                    class_name = results.names[int(cls_idx)]
                    confidence = float(conf)
                    bbox = (int(x1), int(y1), int(x2), int(y2))
                    
                    predictions.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': bbox
                    })
    except Exception as e:
        pass
    
    return predictions

# ============================================================
# MAIN APP
# ============================================================

def main():
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    pTime = 0
    frame_count = 0
    
    print("\n" + "="*70)
    print("SIGN LANGUAGE RECOGNITION APP - YOLOv5")
    print("="*70)
    print("Press 'q' to exit")
    print("="*70 + "\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to read from camera")
            break
        
        frame_count += 1
        
        # ============================================================
        # DETECTION
        # ============================================================
        
        results, annotated_frame = detect_sign_language(frame, model)
        predictions = extract_predictions(results, CONFIDENCE_THRESHOLD)
        
        # ============================================================
        # DISPLAY RESULTS
        # ============================================================
        
        # Show detections on console (every 30 frames)
        if frame_count % 30 == 0:
            print(f"\n{'='*50}")
            print(f"Frame {frame_count}")
            print(f"{'='*50}")
            
            if len(predictions) > 0:
                for i, pred in enumerate(predictions, 1):
                    print(f"  {i}. {pred['class']}: {pred['confidence']*100:.1f}%")
            else:
                print("  No sign language detected")
        
        # ============================================================
        # FPS CALCULATION
        # ============================================================
        
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        
        # ✅ Κάνε το frame writable πριν βάλεις text
        annotated_frame = np.ascontiguousarray(annotated_frame)
        
        # Add FPS to frame
        cv2.putText(
            annotated_frame,
            f"FPS: {int(fps)}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2
        )
        
        # Add detected count
        cv2.putText(
            annotated_frame,
            f"Detections: {len(predictions)}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2
        )
        
        # Add predictions text
        y_offset = 140
        for pred in predictions:
            text = f"{pred['class']}: {pred['confidence']*100:.1f}%"
            cv2.putText(
                annotated_frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            y_offset += 40
        
        # ============================================================
        # SHOW FRAME
        # ============================================================
        
        cv2.imshow('Sign Language Recognition - YOLOv5', annotated_frame)
        
        # ============================================================
        # EXIT
        # ============================================================
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("\n" + "="*70)
            print("Exiting...")
            print("="*70)
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("App closed successfully")

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()


# ============================================================
# ΣΗΜΑΝΤΙΚΑ ΣΗΜΕΙΑ
# ============================================================

"""
1. ✅ annotated_frame.copy() - κάνει το frame writable
2. ✅ np.ascontiguousarray() - εναλλακτικά fix για read-only
3. ✅ model.conf - ορίστε confidence threshold
4. ✅ model.names - παίρνου τα ονόματα των κλάσεων
5. ✅ κάθε 30 frames εκτυπώνει τα αποτελέσματα
"""


# ============================================================
# TROUBLESHOOTING
# ============================================================

"""
ERROR: "Overload resolution failed" - img marked as readonly
FIX: annotated_frame = annotated_frame.copy()

ERROR: FutureWarning about torch.cuda.amp.autocast
FIX: Αυτό είναι απλώς warning, όχι error. Δεν χρειάζεται κάτι.

ERROR: Low FPS
FIX1: model.cpu() - χρησιμοποίησε CPU
FIX2: Χαμηλώστε ανάλυση κάμερας σε 640x480
FIX3: Χρησιμοποίησε GPU αν έχεις NVIDIA card

ERROR: Model not loading
FIX: Βεβαιώσου ότι το best.pt είναι σωστό YOLOv5 model
"""