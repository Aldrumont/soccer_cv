import torch
from PIL import Image
import cv2
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to perform object detection
def detect_objects(image_path):
    # Load image
    img = Image.open(image_path)

    # Perform inference
    results = model(img)

    # Display results
    results.show()

    # Print results
    print(results.xyxy[0])  # bounding boxes and scores
    print(results.names)    # class names

    return results

# Function to draw bounding boxes on the image
def draw_bounding_boxes(image_path, results):
    # Load image
    img = cv2.imread(image_path)

    # Get bounding boxes and labels
    boxes = results.xyxy[0].cpu().numpy()  # convert to numpy array
    labels = results.names

    # Draw bounding boxes and labels
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        label = f"{labels[int(cls)]} {conf:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the image with bounding boxes
    output_path = 'output.jpg'
    cv2.imwrite(output_path, img)
    print(f"Saved output image with bounding boxes to {output_path}")

if __name__ == "__main__":
    # Path to the image
    image_path = 'Screenshot from 2024-06-26 20-53-16.png'

    # Perform object detection
    results = detect_objects(image_path)

    # Draw bounding boxes on the image
    draw_bounding_boxes(image_path, results)
