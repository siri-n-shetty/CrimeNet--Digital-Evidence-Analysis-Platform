import cv2
from ultralytics import YOLO

def detect_assets(image_path):
    model = YOLO('yolov8n.pt')  # Use yolov8m.pt or yolov8l.pt for better accuracy if you have GPU
    asset_classes = ['handbag', 'wallet', 'watch', 'suitcase']  # Add more if needed
    image = cv2.imread(image_path)
    if image is None:
        return {"error": f"Image {image_path} not found."}

    results = model(image)
    detected = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            if cls_name in asset_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected.append({
                    "class": cls_name,
                    "box": [x1, y1, x2, y2]
                })
    return {"assets": detected}

