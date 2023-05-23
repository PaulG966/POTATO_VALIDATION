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
#device 0 model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/modele_boitenoire/weights/last.pt', device='cpu')

# Paramètres pour le calibrage de la caméra
left_crop = 200
right_crop = 150
center_image = 2

camera_id = 2
cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

nb_eje = 6
vector = np.zeros(nb_eje)

count = 0
while True:
    vector.fill(0)
    ret, frame = cap.read()
    
    # Prétraitement de l'image
    frame = frame[:, left_crop:frame.shape[1] - right_crop, :]
    #frame = frame[int(frame.shape[0] / center_image):, :]
    
    # Conversion du frame en une image PIL
    image = Image.fromarray(frame)
    
    
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