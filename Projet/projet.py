import cv2 as cv
import numpy as np
import random




# Kernel pour effet sépia
sepia_kernel = np.array([[0.272, 0.534, 0.131],
                         [0.349, 0.686, 0.168],
                         [0.393, 0.769, 0.189]])


# Redimensionner la taille du flocon
def load_snowflake_image(path, size=(30, 30)):
    snowflake = cv.imread(path, cv.IMREAD_UNCHANGED)
    snowflake = cv.resize(snowflake, size)
    return snowflake


# Chargement de la webcam et des images, cascades..
cap = cv.VideoCapture(0)
snowflake_img = load_snowflake_image('C:/Users/huber/Desktop/TD_Images/Projet/images/snowflakes.png', size=(30, 30))
face_cascade = cv.CascadeClassifier('C:/Users/huber/Desktop/TD_Images/Projet/images/haarcascade_frontalface_alt.xml')


def generate_snowflakes(frame, num_flakes=50):
    height, width, _ = frame.shape
    snowflakes = []
    for _ in range(num_flakes):
        x = random.randint(0, width)
        y = random.randint(0, height)
        snowflakes.append([x, y, random.choice([-1, 1]), random.randint(2, 5)])  # Add direction and speed
    return snowflakes

def move_snowflakes(snowflakes, height, width):
    for flake in snowflakes:
        flake[1] += flake[3]  # Move down with speed
        flake[0] += flake[2]  # Move left or right
        if flake[1] > height:  # Reset when out of frame
            flake[1] = 0
            flake[0] = random.randint(0, width)
        if flake[0] < 0 or flake[0] > width:  # Reset horizontal position
            flake[0] = random.randint(0, width)

def overlay_image_alpha(img, img_overlay, x, y):
    overlay_h, overlay_w = img_overlay.shape[:2]
    if x >= img.shape[1] or y >= img.shape[0]:
        return img
    
    y1, y2 = max(0, y), min(img.shape[0], y + overlay_h)
    x1, x2 = max(0, x), min(img.shape[1], x + overlay_w)
    
    overlay = img_overlay[0:y2-y1, 0:x2-x1]
    alpha_mask = overlay[:, :, 3] / 255.0

    for c in range(0, 3):
        img[y1:y2, x1:x2, c] = (1 - alpha_mask) * img[y1:y2, x1:x2, c] + alpha_mask * overlay[:, :, c]

    return img

def detect_faces(frame, face_cascade):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    enlarged_faces = []
    for (x, y, w, h) in faces:
        enlarged_faces.append((x - 50, y - 70, w + 100, h + 140))  # Région du visage
    return enlarged_faces


def imageFond():
    snowflakes = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        height, width, _ = frame.shape
        if snowflakes is None:
            snowflakes = generate_snowflakes(frame)
        
        faces = detect_faces(frame, face_cascade)
        move_snowflakes(snowflakes, height, width)

        for flake in snowflakes:
            x, y = flake[0], flake[1]
            if not any((x > fx and x < fx + fw and y > fy and y < fy + fh) for fx, fy, fw, fh in faces):
                overlay_image_alpha(frame, snowflake_img, x, y)

        cv.imshow('Fond Neige Interactive', frame)

        # Quitter avec 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def apply_filter(frame, filtre_choisi):
    if filtre_choisi == 1:  # Sans modif
        return frame
    elif filtre_choisi == 2:  # Niveaux de gris
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        return cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    elif filtre_choisi == 3:  # Image binaire
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)
        return cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    elif filtre_choisi == 4:  # Canal vert
        frame[:, :, 0] = 0
        frame[:, :, 2] = 0
        return frame
    elif filtre_choisi == 5:  # Filtre sépia
        sepia = cv.transform(frame, sepia_kernel)
        return np.clip(sepia, 0, 255).astype(np.uint8)
    elif filtre_choisi == 6:  # Effet crayon
        gray, sketch = cv.pencilSketch(frame, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
        return sketch
    else:
        return frame


def imageFondEtFiltre():
    snowflakes = None
    while True:
        try:
            filtre_choisi = int(input("Choisissez un filtre (1 à 6) pour appliquer avec les flocons : "))
            if 1 <= filtre_choisi <= 6:
                break
            else:
                print("Choix invalide. Veuillez entrer un nombre entre 1 et 6.")
        except ValueError:
            print("Entrée invalide. Veuillez entrer un nombre entier entre 1 et 6.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        height, width, _ = frame.shape
        if snowflakes is None:
            snowflakes = generate_snowflakes(frame)
        
        faces = detect_faces(frame, face_cascade)
        move_snowflakes(snowflakes, height, width)

        for flake in snowflakes:
            x, y = flake[0], flake[1]
            if not any((x > fx and x < fx + fw and y > fy and y < fy + fh) for fx, fy, fw, fh in faces):
                overlay_image_alpha(frame, snowflake_img, x, y)

        frame = apply_filter(frame, filtre_choisi)

        cv.imshow('Flocons et Filtre', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def filtreImageEntiere():

    # Message d'erreur en cas de mauvais choix ou mauvaise entrée 
    while True:
        try:
            filtre_choisi = int(input("Choisissez un filtre (1 à 6) pour appliquer avec les flocons : "))
            if 1 <= filtre_choisi <= 6:
                break
            else:
                print("Choix invalide. Veuillez entrer un nombre entre 1 et 6.")
        except ValueError:
            print("Entrée invalide. Veuillez entrer un nombre entier entre 1 et 6.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        frame = apply_filter(frame, filtre_choisi)

        cv.imshow('Filtre Image', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    while True:
        choix = input("Entrez 1 pour appliquer des filtres : 2 pour un fond interactif ou 3 pour les deux et 4 pour quitter : ")
        if choix == "1":
            filtreImageEntiere()
        elif choix == "2":
            imageFond()
        elif choix == "3":
            imageFondEtFiltre()
        elif choix == "4":
            break
        else:
            print("Choix invalide.")

