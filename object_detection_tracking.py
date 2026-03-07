import cv2
from ultralytics import YOLO

# Replace with YOUR phone IP
phone_ip = "192.168.10.8"  # CHANGE THIS
port = "4747"

# DroidCam video stream URL
url = f"http://{phone_ip}:{port}/video"

print("Connecting to:", url)

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("❌ Could not connect to DroidCam stream")
    exit()

print("✅ Connected to DroidCam")

model = YOLO("yolov8n.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not received")
        break

    results = model.track(frame, persist=True, conf=0.4)
    annotated = results[0].plot()

    cv2.imshow("Live Detection & Tracking", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()