import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

model_path = os.path.join(os.path.dirname(__file__), 'models', 'weapons.pt')
weapon_model = YOLO(model_path)

def detect_weapons(image_input):
    """
    Detect weapons in an image.
    Args:
        image_input: PIL.Image object or string path to image
    Returns:
        processed_image (PIL.Image): Image with bounding boxes drawn
        results_dict (dict): Detection results
    """
    # Handle both file paths and PIL.Image
    if isinstance(image_input, str):
        image_pil = Image.open(image_input).convert("RGB")
    else:
        image_pil = image_input.convert("RGB")

    # Convert to OpenCV format
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # Run YOLO
    results = weapon_model(image)

    detections = []
    weapon_detected = False

    for result in results:
        classes = result.names
        cls = result.boxes.cls
        conf = result.boxes.conf
        detections_xyxy = result.boxes.xyxy

        for pos, detection in enumerate(detections_xyxy):
            confidence = float(conf[pos])
            if confidence >= 0.5:
                label_name = classes[int(cls[pos])].lower()

                if any(x in label_name for x in ["gun", "knife"]):
                    weapon_detected = True

                xmin, ymin, xmax, ymax = map(int, detection.tolist())
                label = f"{classes[int(cls[pos])]} {confidence:.2f}"
                color = (0, int(cls[pos]) * 40 % 255, 255)

                # Draw bounding box + label
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(image, label, (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                detections.append({
                    "label": label_name,
                    "confidence": round(confidence, 2),
                    "bbox": [xmin, ymin, xmax, ymax]
                })

    # Convert back to PIL for return
    # processed_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    results_dict = {
        "category": "weapons",
        "weapon_detected": weapon_detected,
        "detections": detections
    }

    return results_dict
