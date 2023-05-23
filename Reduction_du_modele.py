import torch
import numpy as np
import cv2
from PIL import Image
import requests

def binary_to_decimal(vector):
    binary_str = ''.join(str(int(bit)) for bit in vector)
    decimal = int(binary_str, 2)
    return decimal

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp7/weights/last.pt', device='cpu')

camera_id = 1

cap = cv2.VideoCapture(camera_id)

if not cap.isOpened():
    print("Erreur lors de l'ouverture de la caméra")
    exit()

nb_eje = 6
vector = np.zeros(nb_eje)

count = 0

while True:
    vector.fill(0)
    ret, frame = cap.read()

    image = Image.fromarray(frame)

    results = model(image)

    detections = results.pandas().xyxy[0]
    for _, data in detections.iterrows():
        class_name = model.names[int(data[5])]

        x1, y1, x2, y2 = int(data[0]), int(data[1]), int(data[2]), int(data[3])

        for i in range(nb_eje):
            if (frame.shape[1] / nb_eje) * i < x1 < (frame.shape[1] / nb_eje) * (i + 1):
                vector[i] = 1 if class_name == 'Fresh' else vector[i]
            elif (frame.shape[1] / nb_eje) * i < x2 < (frame.shape[1] / nb_eje) * (i + 1):
                vector[i] = 1 if class_name == 'Fresh' else vector[i]
            elif (frame.shape[1] / nb_eje) * i < (x1 + x2) / 2 < (frame.shape[1] / nb_eje) * (i + 1):
                vector[i] = 1 if class_name == 'Fresh' else vector[i]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f'{class_name}'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    count += 1

    variable_PU = binary_to_decimal(vector)
    print("n°",count, variable_PU)

    for i in range(1, nb_eje):
        x = int((frame.shape[1] / nb_eje) * i)
        cv2.line(frame, (x, 0), (x, frame.shape[0]), (0, 255, 0), 2)

    cv2.imshow('YOLO', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
