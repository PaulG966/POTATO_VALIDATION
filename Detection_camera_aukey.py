import torch
import numpy as np
import cv2
from PIL import Image
import os 

os.chdir("yolov5")


#chargement du modele :
model2 = torch.hub.load('ultralytics/yolov5', 'custom',path='runs/train/exp7/weights/last.pt') 

# On considère qu'il y a 6 servo-moteurs pour les éjections
nb_eje = 6
vector = np.zeros(nb_eje)

cap = cv2.VideoCapture(2)



###                                          CALIBRAGE CAMERA AUKEY

Enlever_sur_cote_gauche= 33
Enlever_sur_cote_droit= 155
Centrer_image= 4    # Coupe l'image en longueur 

###                                           FIN CALIBRAGE
count=0
while True:
    vector = np.zeros(nb_eje)
    ret, frame = cap.read()
    
    # Supprimer 213 pixels en largeur à gauche et à droite
    width = frame.shape[1]    
    frame = frame[:, Enlever_sur_cote_gauche:width-Enlever_sur_cote_droit, :]
    
    # Couper la partie supérieure de l'image
    height = frame.shape[0]
    frame = frame[int(height/Centrer_image):, :]
       
   
    
    # Tracer des droites verticales
  
    for i in range(1, nb_eje):
        x = int((frame.shape[1]  / 6)*i)
        cv2.line(frame, (x, 0), (x, frame.shape[0]), (0, 255, 0), 2)
        
    
    # Convertir le frame capturé en une image PIL
    image = Image.fromarray(frame)
    
    # Effectuer la détection d'objet
    results = model2(image)
    
    for detection in results.pandas().xyxy[0].iterrows():
        index, data = detection
        class_name = model2.names[int(data[5])]
        
        # Ajuster les coordonnées en fonction du redimensionnement de l'image
        x = (data[0] + data[2]) / 2
        y = (data[1] + data[3]) / 2
        #print("Objet détecté : classe={}, position=({}, {})".format(class_name, x, y))
        
        # Remplissage du vecteur :
        if class_name == 'Fresh':
            for i in range(nb_eje):
                if (frame.shape[1] / nb_eje) * i < x < (frame.shape[1] / nb_eje) * (i+1):
                    vector[i] = 1
                elif (frame.shape[1] / nb_eje) * i < data[0] < (frame.shape[1] / nb_eje) * (i+1):
                    vector[i] = 1
                elif (frame.shape[1] / nb_eje) * i < data[2] < (frame.shape[1] / nb_eje) * (i+1):
                    vector[i] = 1
    count +=1
    print(count," ",vector)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
