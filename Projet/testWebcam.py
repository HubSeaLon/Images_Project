import cv2

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

# Charger les classificateurs Haar pour les visages et nez
face_cascade = cv2.CascadeClassifier('C:/Users/timot/OneDrive/Desktop/Travail/Code/traitementImage/TD6/haarcascade_frontalface_alt.xml')
nose_cascade = cv2.CascadeClassifier('C:/Users/timot/OneDrive/Desktop/Travail/Code/traitementImage/TD6/haarcascade_mcs_nose.xml')

# Charger les images pour les oreilles et le nez (avec transparence)
oreilleChien = cv2.imread('C:/Users/timot/OneDrive/Desktop/Travail/Code/traitementImage/TD6/oreilleChien2-removebg-preview.png', cv2.IMREAD_UNCHANGED)
nezChien = cv2.imread('C:/Users/timot/OneDrive/Desktop/Travail/Code/traitementImage/TD6/nezChien2-removebg-preview.png', cv2.IMREAD_UNCHANGED)

# Fonction pour superposer une image avec transparence
def overlay_image(background, overlay, x, y, w, h):
    overlay_resized = cv2.resize(overlay, (w, h))  # Redimensionner l'image
    for i in range(overlay_resized.shape[0]):
        for j in range(overlay_resized.shape[1]):
            if overlay_resized.shape[2] == 4 and overlay_resized[i, j][3] > 0:  # Vérifie la transparence
                background[y + i, x + j] = overlay_resized[i, j][:3]

# Fonction pour ajouter les oreilles
def add_ears(frame, gray, faces):
    for (x, y, w, h) in faces:
        ears_width = int(w * 1.5)
        ears_height = int(h * 0.6)
        ears_x = x - int(w * 0.25)
        ears_y = y - int(h * 0.5)
        overlay_image(frame, oreilleChien, ears_x, ears_y, ears_width, ears_height)

# Fonction pour ajouter le nez
def add_nose(frame, gray, faces):
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        noses = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=5)
        for (nx, ny, nw, nh) in noses:
            nose_width = int(nw * 3.5)
            nose_height = int(nh * 2.3)
            nose_x = x + nx - (nose_width - nw) // 2
            nose_y = y + ny - (nose_height - nh) // 2 + 10
            overlay_image(frame, nezChien, nose_x, nose_y, nose_width, nose_height)
            break  # Arrêter après le premier nez trouvé

# Fonction pour ajouter les oreilles et le nez
def add_ears_and_nose(frame, gray, faces):
    add_ears(frame, gray, faces)
    add_nose(frame, gray, faces)

# Vérifier si la webcam est ouverte
if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la webcam.")
    exit()

print("Appuie sur 'q' pour quitter.")
print("Appuie sur '1' pour ajouter les oreilles.")
print("Appuie sur '2' pour ajouter le nez.")
print("Appuie sur '3' pour ajouter les deux filtres.")

# Définir le filtre actif (par défaut aucun filtre)
active_filter = None

# Boucle principale
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire le flux vidéo.")
        break

    # Convertir en niveaux de gris pour la détection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détection des visages
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Appliquer le filtre en fonction de l'entrée utilisateur
    if active_filter == '1':
        add_ears(frame, gray, faces)
    elif active_filter == '2':
        add_nose(frame, gray, faces)
    elif active_filter == '3':
        add_ears_and_nose(frame, gray, faces)

    # Afficher le résultat
    cv2.imshow("Webcam - Filtres de Chien", frame)

    # Gérer les entrées utilisateur
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quitter avec 'q'
        break
    elif key == ord('1'):  # Activer le filtre des oreilles
        active_filter = '1'
        print("Filtre activé : Oreilles uniquement.")
    elif key == ord('2'):  # Activer le filtre du nez
        active_filter = '2'
        print("Filtre activé : Nez uniquement.")
    elif key == ord('3'):  # Activer les deux filtres
        active_filter = '3'
        print("Filtre activé : Oreilles et Nez.")

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
