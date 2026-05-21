import cv2
import os
import numpy as np

DOSSIER_PHOTOS = r"C:\Users\pc\Desktop\RCP_simulator\nn\photo"
DOSSIER_LABELS = r"C:\Users\pc\Desktop\RCP_simulator\nn\labels_portes"

os.makedirs(DOSSIER_LABELS, exist_ok=True)

photos = [f for f in os.listdir(DOSSIER_PHOTOS)
          if f.lower().endswith((".jpg", ".jpeg", ".png"))]

print(f"{len(photos)} photos trouvees")

# Variables globales
rectangles = []
dessin     = False
x_start    = y_start = x_curr = y_curr = 0
img_affich = None
NOM_FENETRE = "LNE Annotateur"

def rafraichir():
    global img_affich, rectangles, x_start, y_start, x_curr, y_curr, dessin
    img_temp = img_affich.copy()

    # Dessiner rectangles sauvegardés
    for rx1, ry1, rx2, ry2 in rectangles:
        cv2.rectangle(img_temp, (rx1, ry1), (rx2, ry2), (255, 0, 255), 2)
        cv2.putText(img_temp, "door",
                    (rx1, ry1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    # Rectangle en cours
    if dessin:
        cv2.rectangle(img_temp,
                      (x_start, y_start),
                      (x_curr, y_curr),
                      (0, 255, 0), 2)

    # Infos
    h = img_temp.shape[0]
    cv2.putText(img_temp, f"Portes : {len(rectangles)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2)
    cv2.putText(img_temp,
                "CLIC+GLISSER:tracer | D:suivant | A:prec | S:sauver | Z:effacer | Q:quitter",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imshow(NOM_FENETRE, img_temp)

def souris(event, x, y, flags, param):
    global dessin, x_start, y_start, x_curr, y_curr, rectangles

    if event == cv2.EVENT_LBUTTONDOWN:
        dessin  = True
        x_start = x
        y_start = y
        x_curr  = x
        y_curr  = y

    elif event == cv2.EVENT_MOUSEMOVE:
        x_curr = x
        y_curr = y
        if dessin:
            rafraichir()

    elif event == cv2.EVENT_LBUTTONUP:
        if dessin:
            dessin = False
            x_curr = x
            y_curr = y
            # Ajouter seulement si rectangle assez grand
            if abs(x - x_start) > 10 and abs(y - y_start) > 10:
                rectangles.append((
                    min(x_start, x), min(y_start, y),
                    max(x_start, x), max(y_start, y)
                ))
            rafraichir()

def sauver_labels(nom_photo, rects, img_w, img_h):
    nom_base = os.path.splitext(nom_photo)[0]
    chemin   = os.path.join(DOSSIER_LABELS, nom_base + ".txt")
    with open(chemin, "w") as f:
        for rx1, ry1, rx2, ry2 in rects:
            cx = ((rx1 + rx2) / 2) / img_w
            cy = ((ry1 + ry2) / 2) / img_h
            w  = (rx2 - rx1) / img_w
            h  = (ry2 - ry1) / img_h
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    print(f"Sauvegarde : {len(rects)} portes — {chemin}")

def charger_photo(idx):
    global img_affich, rectangles
    nom_photo = photos[idx]
    chemin    = os.path.join(DOSSIER_PHOTOS, nom_photo)
    img       = cv2.imread(chemin)

    if img is None:
        return False

    # Redimensionner
    h, w   = img.shape[:2]
    scale  = min(1200/w, 800/h, 1.0)
    img_affich = cv2.resize(img, (int(w*scale), int(h*scale)))

    # Charger labels existants
    rectangles = []
    nom_base   = os.path.splitext(nom_photo)[0]
    chemin_lbl = os.path.join(DOSSIER_LABELS, nom_base + ".txt")
    if os.path.exists(chemin_lbl):
        ih, iw = img_affich.shape[:2]
        with open(chemin_lbl) as f:
            for ligne in f:
                parts = ligne.strip().split()
                if len(parts) == 5:
                    cx = float(parts[1]) * iw
                    cy = float(parts[2]) * ih
                    bw = float(parts[3]) * iw
                    bh = float(parts[4]) * ih
                    rectangles.append((
                        int(cx-bw/2), int(cy-bh/2),
                        int(cx+bw/2), int(cy+bh/2)
                    ))

    # Titre fenêtre
    cv2.setWindowTitle(NOM_FENETRE,
                       f"LNE Annotateur — {idx+1}/{len(photos)} — {nom_photo}")
    return True

# ── Lancer ───────────────────────────────────────────────
cv2.namedWindow(NOM_FENETRE, cv2.WINDOW_NORMAL)
cv2.resizeWindow(NOM_FENETRE, 1200, 800)
cv2.setMouseCallback(NOM_FENETRE, souris)

idx = 0
charger_photo(idx)
rafraichir()

while True:
    key = cv2.waitKey(20) & 0xFF

    if key == ord('d'):
        ih, iw = img_affich.shape[:2]
        sauver_labels(photos[idx], rectangles, iw, ih)
        idx = min(idx + 1, len(photos) - 1)
        charger_photo(idx)
        rafraichir()

    elif key == ord('a'):
        ih, iw = img_affich.shape[:2]
        sauver_labels(photos[idx], rectangles, iw, ih)
        idx = max(idx - 1, 0)
        charger_photo(idx)
        rafraichir()

    elif key == ord('s'):
        ih, iw = img_affich.shape[:2]
        sauver_labels(photos[idx], rectangles, iw, ih)

    elif key == ord('z'):
        if rectangles:
            rectangles.pop()
        rafraichir()

    elif key == ord('q'):
        ih, iw = img_affich.shape[:2]
        sauver_labels(photos[idx], rectangles, iw, ih)
        break

cv2.destroyAllWindows()
print("Annotation terminee !")