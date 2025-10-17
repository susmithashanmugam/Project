from ultralytics import YOLO
import cv2

# Load your trained model (replace with your actual path)
model = YOLO(r"runs\detect\train5\weights\best.pt")

# Open webcam (0 = default camera, change if needed)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame")
        break

    # Run YOLOv8 prediction
    results = model.predict(frame, conf=0.5, show=False)

    # Draw results on frame
    annotated_frame = results[0].plot()

    # Show live predictions
    cv2.imshow("YOLOv8 Live Detection", annotated_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
