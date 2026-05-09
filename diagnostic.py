# DIAGNOSTIC SCRIPT - CHECK best.pt MODEL
# Τρέξε αυτό για να δεις τι παίζει με το μοντέλο σου

import torch
import os
import cv2
import numpy as np

MODEL_PATH = '/home/charis/Desktop/Projects/Gesture-Recognition-using-video-/app/best.pt'

print("\n" + "="*70)
print("BEST.PT MODEL DIAGNOSTIC")
print("="*70)

# ============================================================
# 1. CHECK FILE
# ============================================================

print("\n1. FILE CHECK:")
print("-" * 70)

if os.path.exists(MODEL_PATH):
    size_mb = os.path.getsize(MODEL_PATH) / (1024*1024)
    print(f"✓ File exists: {MODEL_PATH}")
    print(f"✓ File size: {size_mb:.2f} MB")
    
    if size_mb < 1:
        print("⚠️  WARNING: File is very small (< 1 MB) - might be corrupted")
    elif size_mb > 500:
        print("⚠️  WARNING: File is very large (> 500 MB) - unusual for YOLOv5")
else:
    print(f"✗ FILE NOT FOUND: {MODEL_PATH}")
    exit()

# ============================================================
# 2. LOAD WITH TORCH
# ============================================================

print("\n2. PYTORCH LOAD CHECK:")
print("-" * 70)

try:
    weights = torch.load(MODEL_PATH, map_location='cpu')
    print(f"✓ PyTorch loaded successfully")
    print(f"✓ Type: {type(weights)}")
    
    if isinstance(weights, dict):
        print(f"✓ Keys: {list(weights.keys())[:5]}")
except Exception as e:
    print(f"✗ PyTorch load failed: {e}")

# ============================================================
# 3. LOAD WITH YOLOV5
# ============================================================

print("\n3. YOLOv5 LOAD CHECK:")
print("-" * 70)

try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
    print(f"✓ YOLOv5 model loaded successfully")
    print(f"✓ Model device: {next(model.parameters()).device}")
    print(f"✓ Model classes: {model.names}")
    print(f"✓ Number of classes: {len(model.names)}")
    
    # Check if classes are named correctly
    if len(model.names) == 0:
        print("⚠️  WARNING: No classes found in model!")
    
except Exception as e:
    print(f"✗ YOLOv5 load failed: {e}")
    print("⚠️  This might mean best.pt is corrupted or not a valid YOLOv5 model")
    exit()

# ============================================================
# 4. TEST INFERENCE
# ============================================================

print("\n4. INFERENCE TEST:")
print("-" * 70)

try:
    # Create a dummy image (640x640x3)
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    print(f"Testing with dummy image (640x640x3)...")
    results = model(dummy_img)
    
    print(f"✓ Inference successful")
    print(f"✓ Results type: {type(results)}")
    
    # Check detections
    try:
        detections = results.xyxy[0]
        print(f"✓ Detections shape: {detections.shape}")
        print(f"✓ Detections found: {len(detections)}")
        
        if len(detections) > 0:
            print("⚠️  Dummy image produced detections (might be overfitting)")
        else:
            print("✓ No detections on dummy image (good sign)")
            
    except Exception as e:
        print(f"⚠️  Could not parse detections: {e}")
    
except Exception as e:
    print(f"✗ Inference failed: {e}")
    print("⚠️  Model might not be trained properly")

# ============================================================
# 5. TEST WITH REAL IMAGE
# ============================================================

print("\n5. REAL CAMERA TEST:")
print("-" * 70)

try:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Camera not available")
    else:
        ret, frame = cap.read()
        
        if ret:
            print(f"✓ Camera frame captured: {frame.shape}")
            
            # Run inference
            results = model(frame)
            detections = results.xyxy[0]
            
            print(f"✓ Inference on real frame successful")
            print(f"✓ Detections: {len(detections)}")
            
            if len(detections) > 0:
                print("\nDetection details:")
                for i, det in enumerate(detections):
                    x1, y1, x2, y2, conf, cls_idx = det
                    class_name = model.names[int(cls_idx)]
                    print(f"  {i+1}. {class_name}: {float(conf)*100:.1f}%")
            else:
                print("⚠️  No detections on real frame - model might not be trained")
        else:
            print("✗ Could not read from camera")
        
        cap.release()
        
except Exception as e:
    print(f"✗ Real camera test failed: {e}")

# ============================================================
# 6. SUMMARY & RECOMMENDATIONS
# ============================================================

print("\n" + "="*70)
print("SUMMARY & RECOMMENDATIONS")
print("="*70)

print("""
✓ If all checks passed:
  - Model is valid and can load
  - You can use it for inference
  - Poor results might be due to:
    1. Insufficient training data
    2. Insufficient training epochs
    3. Poor image quality or lighting
    4. Model not converged yet

⚠️  If inference test failed:
  - Model might be corrupted
  - Retrain the model

⚠️  If classes are empty:
  - Model might not have been trained properly
  - Check your training code

NEXT STEPS:
1. Check your training data quality
2. Try training for more epochs
3. Check training metrics (loss, accuracy)
4. Try different augmentation settings
5. Ensure good lighting in your environment
""")

print("="*70 + "\n")