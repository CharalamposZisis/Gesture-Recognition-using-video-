import cv2
import torch
import numpy as np
from pathlib import Path

# ── Load model ──────────────────────────────────────────────────────────────
MODEL_PATH = "best.pt"

print(f"[INFO] Loading model from {MODEL_PATH} ...")
model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=MODEL_PATH,
    force_reload=False,
    trust_repo=True,
)
model.eval()
model.conf = 0.5   # confidence threshold
model.iou  = 0.45  # NMS IoU threshold
print(f"[INFO] Classes: {model.names}")

# ── Webcam ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Check your camera index.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ── Colours per class ────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
COLORS = {
    name: tuple(int(c) for c in rng.integers(80, 230, 3))
    for name in model.names.values()
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.9
THICKNESS  = 2

print("[INFO] Press  Q  to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Empty frame – retrying …")
        continue

    # ── Inference ────────────────────────────────────────────────────────────
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb, size=640)

    # ── Draw detections ──────────────────────────────────────────────────────
    detections = results.xyxy[0].cpu().numpy()   # [x1,y1,x2,y2,conf,cls]

    for *xyxy, conf, cls_id in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        label = model.names[int(cls_id)]
        color = COLORS[label]

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS)

        # Label background
        text      = f"{label}  {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
        cv2.rectangle(frame,
                      (x1, y1 - th - baseline - 6),
                      (x1 + tw + 4, y1),
                      color, -1)

        # Label text
        cv2.putText(frame, text,
                    (x1 + 2, y1 - baseline - 2),
                    FONT, FONT_SCALE,
                    (255, 255, 255), THICKNESS, cv2.LINE_AA)

    # ── HUD ──────────────────────────────────────────────────────────────────
    n = len(detections)
    cv2.putText(frame, f"Detections: {n}",
                (10, 30), FONT, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Press Q to quit",
                (10, 60), FONT, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imshow("Gesture / Letter Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Done.")