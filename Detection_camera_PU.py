import torch
import numpy as np
import cv2
from PIL import Image
import requests



'''
#########   ROS ########
import rospy
from std_msgs.msg import UInt8
import time

def init_high_level_node():
    rospy.init_node('high_level_publisher', anonymous=True)

def publish_cmd_led(value, duration):
    pub = rospy.Publisher('cmd_led', UInt8, queue_size=10)
    rate = rospy.Rate(100)  # 10 Hz
    end_time = time.time() + duration  # Calculate end time

    while not rospy.is_shutdown() and time.time() < end_time:
        cmd = UInt8()
        cmd.data = value

        pub.publish(cmd)
        rate.sleep()
#########   ROS ########

init_high_level_node()

'''

def binary_to_decimal(vector):
    binary_str = ''.join(str(int(bit)) for bit in vector)
    decimal = int(binary_str, 2)
    return decimal


# Chargement du modèle
##device 0
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp7/weights/last.pt', device='cpu')

# Paramètres pour la capture vidéo
camera_id = 2
capture_width = 1280
capture_height = 720
capture_fps = 30

# Paramètres pour le calibrage de la caméra
left_crop =0
right_crop = 180
center_image=1

# Initialisation de la capture vidéo
cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
cap.set(cv2.CAP_PROP_FPS, capture_fps)

nb_eje = 6
vector = np.zeros(nb_eje)

count = 0
while True:
    vector.fill(0)
    ret, frame = cap.read()
    
    # Prétraitement de l'image
    frame = frame[:, left_crop:capture_width - right_crop, :]
    frame = frame[int(frame.shape[0] / center_image):, :]
    
    # Conversion du frame en une image PIL
    image = Image.fromarray(frame)
    
    # Détection d'objets avec le modèle YOLOv5
    results = model(image)
    
    # Analyse des résultats de détection
    detections = results.pandas().xyxy[0]
    for _, data in detections.iterrows():
        class_name = model.names[int(data[5])]
        
        # Calcul des coordonnées ajustées en fonction du redimensionnement de l'image
        x1, y1, x2, y2 = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        
        # Remplissage du vecteur
        for i in range(nb_eje):
            if (frame.shape[1] / nb_eje) * i < x1  < (frame.shape[1] / nb_eje) * (i + 1):
                vector[i] = 1 if class_name == 'Fresh' else vector[i]
            elif (frame.shape[1] / nb_eje) * i < x2  < (frame.shape[1] / nb_eje) * (i + 1):
                vector[i] = 1 if class_name == 'Fresh' else vector[i]
            elif (frame.shape[1] / nb_eje) * i < (x1 +x2)/2  < (frame.shape[1] / nb_eje) * (i + 1):
                vector[i] = 1 if class_name == 'Fresh' else vector[i]
        
        # Dessin du contour de l'objet détecté
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Affichage du nom de l'objet détecté
        label = f'{class_name}'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    count += 1

    
    variable_PU=binary_to_decimal(vector)
   # publish_cmd_led(variable_PU, 0.1)  # Set the value and duration here (0-255, seconds)

    print(count, vector,variable_PU)      
    
    # Tracer des droites vertes
    for i in range(1, nb_eje):
        x = int((frame.shape[1] / nb_eje) * i)
        cv2.line(frame, (x, 0), (x, frame.shape[0]), (0, 255, 0), 2)
    
    # Affichage de l'image avec les résultats de détection
    cv2.imshow('YOLO', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()