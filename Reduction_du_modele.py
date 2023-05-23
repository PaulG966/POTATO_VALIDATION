import cv2

# Paramètres pour la capture vidéo
camera_id = 2
capture_width = 1280
capture_height = 720
capture_fps = 30

# Paramètres pour le calibrage de la caméra
left_crop = 0
right_crop = 180
center_image = 1

# Initialisation de la capture vidéo
cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
cap.set(cv2.CAP_PROP_FPS, capture_fps)

# Vérification si la capture vidéo est ouverte correctement
if not cap.isOpened():
    print("Impossible d'ouvrir la caméra")
    exit()

# Boucle de capture d'images
while True:
    # Lecture de l'image de la caméra
    ret, frame = cap.read()

    # Vérification si la lecture de l'image est réussie
    if not ret:
        print("Échec de la lecture de l'image")
        break

    # Prétraitement de l'image (calibrage de la caméra)
    frame = frame[:, left_crop:capture_width - right_crop, :]
    frame = frame[int(frame.shape[0] / center_image):, :]

    # Vérification de la taille de l'image
    if frame.shape[0] > 0 and frame.shape[1] > 0:
        # Affichage de l'image
        cv2.imshow('Caméra', frame)

    # Attendre l'appui sur la touche 'q' pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libération des ressources
cap.release()
cv2.destroyAllWindows()
