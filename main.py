import cv2
import numpy as np

# Chargement du modèle de détection du visage
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialisation de la caméra
cap = cv2.VideoCapture(0)

while True:
    # Lecture d'une image depuis la caméra
    ret, frame = cap.read()

    # Conversion en niveaux de gris pour la détection du visage
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détection du visage
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Dessin d'un rectangle autour du visage détecté
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Région d'intérêt (ROI) pour les yeux
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Détection des yeux dans la ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # Dessin d'un rectangle autour de chaque œil détecté
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

    # Dessin des lignes reliant les points centraux du visage
    if len(faces) > 0:
        face = faces[0]  # Prend le premier visage détecté
        x, y, w, h = face
        center_x = x + w // 2
        center_y = y + h // 2

        # Dessin des lignes vertes reliant les points centraux des zones du visage
        cv2.line(frame, (center_x, center_y), (center_x, y), (0, 255, 0), 2)  # Ligne vers le menton
        cv2.line(frame, (center_x, center_y), (x, y + h // 2), (0, 255, 0), 2)  # Ligne vers le nez
        cv2.line(frame, (center_x, center_y), (x + w, y + h // 2), (0, 255, 0), 2)  # Ligne vers le nez
        cv2.line(frame, (center_x, center_y), (x + w // 2, y + h), (0, 255, 0), 2)  # Ligne vers la bouche

    # Affichage de la vidéo en direct
    cv2.imshow('Facial Recognition', frame)

    # Sortir de la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture vidéo et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
