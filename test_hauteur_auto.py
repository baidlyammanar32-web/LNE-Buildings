import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO
import os

modele_yolo  = YOLO("yolov8n.pt")
modele_portes = YOLO(r"C:\Users\pc\Desktop\RCP_simulator\nn\runs\detect\modele_portes\weights\best.pt")

# Variables globales
points       = []
img_affich   = None
pixels_ref   = None
hauteur_ref  = None
NOM_FENETRE  = "LNE — Clic Haut puis Bas du batiment"

def souris(event, x, y, flags, param):
    global points, img_affich
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))
            couleur = (255, 0, 0) if len(points) == 1 else (0, 200, 0)
            label   = "Haut" if len(points) == 1 else "Sol"
            cv2.circle(img_affich, (x, y), 6, couleur, -1)
            cv2.putText(img_affich, label, (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, couleur, 2)
            if len(points) == 2:
                # Dessiner ligne
                cv2.line(img_affich,
                         (img_affich.shape[1]//2, points[0][1]),
                         (img_affich.shape[1]//2, points[1][1]),
                         (0, 255, 255), 2)
                px_bat = abs(points[1][1] - points[0][1])
                if pixels_ref and hauteur_ref:
                    h_m = (px_bat / pixels_ref) * hauteur_ref
                    cv2.putText(img_affich,
                                f"{px_bat}px = {h_m:.2f} m",
                                (img_affich.shape[1]//2 + 10,
                                 (points[0][1] + points[1][1])//2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 255, 255), 2)
                    print(f"\nHauteur estimee : {h_m:.2f} m")
                    print(f"Pixels batiment : {px_bat} px")
            cv2.imshow(NOM_FENETRE, img_affich)

def analyser_photo(chemin_photo):
    global img_affich, points, pixels_ref, hauteur_ref

    points = []

    # Charger image
    image = Image.open(chemin_photo).convert("RGB")
    image = ImageOps.exif_transpose(image)

    scale   = min(900 / image.width, 700 / image.height, 1.0)
    largeur = int(image.width  * scale)
    hauteur = int(image.height * scale)
    image_r = image.resize((largeur, hauteur))

    img_np  = np.array(image_r)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # ── Détection YOLO voitures ──────────────────────────
    resultats = modele_yolo(img_bgr, verbose=False)
    pixels_ref  = None
    hauteur_ref = None

    for r in resultats:
        for box in r.boxes:
            if modele_yolo.names[int(box.cls)] == "car":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                px = y2 - y1
                cv2.rectangle(img_bgr, (x1,y1), (x2,y2), (0,200,0), 2)
                cv2.putText(img_bgr, f"Voiture {px}px",
                            (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 2)
                if pixels_ref is None:
                    pixels_ref  = px
                    hauteur_ref = 1.5

    # ── Détection portes ─────────────────────────────────
    resultats_p = modele_portes(img_bgr, verbose=False, conf=0.3)
    for r in resultats_p:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            px = y2 - y1
            cv2.rectangle(img_bgr, (x1,y1), (x2,y2), (255,0,255), 2)
            cv2.putText(img_bgr, f"Porte {px}px",
                        (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
            if pixels_ref is None:
                pixels_ref  = px
                hauteur_ref = 2.2

    # Afficher référence trouvée
    if pixels_ref:
        ref_label = f"Ref auto : {pixels_ref}px = {hauteur_ref}m"
        print(f"Reference : {ref_label}")
    else:
        print("Aucune reference detectee — entre manuellement")
        pixels_ref  = int(input("Pixels reference : "))
        hauteur_ref = float(input("Hauteur reelle (m) : "))

    # Instructions
    cv2.putText(img_bgr,
                "CLIC 1 = Haut batiment | CLIC 2 = Sol | R = Reset | Q = Quitter",
                (10, img_bgr.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    img_affich = img_bgr.copy()

    cv2.namedWindow(NOM_FENETRE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(NOM_FENETRE, largeur, hauteur)
    cv2.setMouseCallback(NOM_FENETRE, souris)
    cv2.imshow(NOM_FENETRE, img_affich)

    hauteur_finale = None
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset
            points     = []
            img_affich = img_bgr.copy()
            cv2.putText(img_affich,
                        "CLIC 1 = Haut batiment | CLIC 2 = Sol | R = Reset | Q = Quitter",
                        (10, img_affich.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.imshow(NOM_FENETRE, img_affich)

        if len(points) == 2 and pixels_ref:
            px_bat = abs(points[1][1] - points[0][1])
            hauteur_finale = (px_bat / pixels_ref) * hauteur_ref

    cv2.destroyAllWindows()
    return hauteur_finale

# ── Test ─────────────────────────────────────────────────
dossier = r"C:\Users\pc\Desktop\RCP_simulator\nn\photo"
photos  = [f for f in os.listdir(dossier) if f.endswith(".jpg")]

for nom in photos:
    chemin = os.path.join(dossier, nom)
    print(f"\nPhoto : {nom}")
    h = analyser_photo(chemin)
    if h:
        print(f"Hauteur finale : {h:.2f} m")
    rep = input("Photo suivante ? (o/n) : ")
    if rep.lower() != 'o':
        break