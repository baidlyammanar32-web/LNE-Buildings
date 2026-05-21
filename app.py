import streamlit as st
from PIL import Image , ImageOps
import exifread
import folium
import math
import cv2
import numpy as np
from streamlit.components.v1 import html
from ultralytics import YOLO
from streamlit_drawable_canvas import st_canvas
from datetime import datetime
from transformers import pipeline
import io, base64

from torchvision import transforms, models
import torch
import torch.nn as nn


@st.cache_resource
def charger_modele_rotation():
    MODELE_PATH = r"C:\Users\pc\Desktop\RCP_simulator\nn\modele_rotation.pth"
    modele = models.resnet18(weights=None)
    modele.fc = nn.Linear(modele.fc.in_features, 4)
    modele.load_state_dict(torch.load(MODELE_PATH, map_location="cpu"))
    modele.eval()
    return modele


def predire_rotation(image_pil):
    """Ton propre modèle IA prédit l'orientation"""
    modele = charger_modele_rotation()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Classes dans l'ordre du dataset
    classes = ["0", "180", "270", "90"]

    img_tensor = transform(image_pil.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        output = modele(img_tensor)
        _, predicted = output.max(1)
        classe = classes[predicted.item()]

    return int(classe)
def detecter_ciel_et_corriger(image_pil):
    """Détecte le ciel et corrige la rotation — le ciel doit être en haut"""
    try:
        image_np = np.array(image_pil.resize((224, 224)))

        def score_ciel(img):
            # Le ciel = zone claire + bleutée en haut
            haut   = img[:56, :, :]   # 25% du haut
            bas    = img[168:, :, :]  # 25% du bas

            # Score bleu/blanc dans la zone
            def score_zone(zone):
                r, g, b = zone[:,:,0], zone[:,:,1], zone[:,:,2]
                # Ciel = bleu > rouge ET luminosité haute
                bleu  = (b.astype(int) - r.astype(int)).mean()
                lumin = zone.mean()
                return bleu + lumin * 0.3

            return score_zone(haut) - score_zone(bas)

        # Tester les 4 orientations
        scores = {}
        for angle in [0, 90, 180, 270]:
            img_rot = np.array(image_pil.rotate(angle, expand=True).resize((224, 224)))
            scores[angle] = score_ciel(img_rot)

        # L'angle avec le meilleur score = ciel en haut
        meilleur = max(scores, key=scores.get)
        print(f"Scores ciel : {scores}")
        print(f"Meilleur angle : {meilleur}°")
        return meilleur

    except Exception as e:
        print(f"Erreur détection ciel : {e}")
        return 0

def corriger_rotation_auto(image_pil, idx):
    """Corrige automatiquement avec détection du ciel"""

    # Rotation manuelle prioritaire
    angle_manuel = st.session_state.get(f"rotation_{idx}", None)
    if angle_manuel is not None:
        return image_pil.rotate(angle_manuel, expand=True)

    # Détection automatique (mise en cache)
    key = f"angle_auto_{idx}"
    # Toujours recalculer — pas de cache
    angle = detecter_ciel_et_corriger(image_pil)
    st.session_state[key] = angle

    angle = st.session_state[key]
    if angle != 0:
        return image_pil.rotate(angle, expand=True)
    return image_pil
st.set_page_config(layout="wide")
@st.cache_resource
def charger_modele():
    return YOLO("yolov8n.pt")

def extraire_gps(photo):
    tags = exifread.process_file(photo, details=False)
    try:
        lat = tags["GPS GPSLatitude"].values
        lat_ref = tags["GPS GPSLatitudeRef"].values
        lon = tags["GPS GPSLongitude"].values
        lon_ref = tags["GPS GPSLongitudeRef"].values
        def convertir(valeur):
            d = float(valeur[0].num) / float(valeur[0].den)
            m = float(valeur[1].num) / float(valeur[1].den)
            s = float(valeur[2].num) / float(valeur[2].den)
            return d + m/60 + s/3600
        latitude  = convertir(lat)
        longitude = convertir(lon)
        if lat_ref != "N": latitude  = -latitude
        if lon_ref != "E": longitude = -longitude
        return latitude, longitude
    except:
        return None, None

def extraire_azimut(photo):
    tags = exifread.process_file(photo, details=False)
    try:
        az = tags["GPS GPSImgDirection"].values[0]
        return float(az.num) / float(az.den)
    except:
        return None

def extraire_date(photo):
    tags = exifread.process_file(photo, details=False)
    try:
        date_str = str(tags["EXIF DateTimeOriginal"])
        dt = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
        return dt.strftime("%d-%b-%Y")
    except:
        return datetime.now().strftime("%d-%b-%Y")

def afficher_carte_une_photo(d):
    if not d["lat"] or not d["lon"]:
        return None
    carte = folium.Map(location=[d["lat"], d["lon"]], zoom_start=18)
    folium.Marker(
        [d["lat"], d["lon"]],
        popup=f"{d['nom']}<br>Azimut: {d['azimut']}°" if d["azimut"] else d["nom"],
        tooltip="📷 Position photo",
        icon=folium.Icon(color="red", icon="camera", prefix="fa")
    ).add_to(carte)
    if d["azimut"]:
        distance = 0.0004
        lat2 = d["lat"] + distance * math.cos(math.radians(d["azimut"]))
        lon2 = d["lon"] + distance * math.sin(math.radians(d["azimut"]))
        folium.PolyLine(
            [[d["lat"], d["lon"]], [lat2, lon2]],
            color="blue", weight=3,
            tooltip=f"Azimut: {d['azimut']:.1f}°"
        ).add_to(carte)
    return carte._repr_html_()

MODELE_PORTES_PATH = r"C:\Users\pc\Desktop\RCP_simulator\nn\runs\detect\modele_portes\weights\best.pt"

@st.cache_resource
def charger_modele_portes():
    return YOLO(MODELE_PORTES_PATH)

def detecter_objets(image_pil, modele):
    image_np  = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # ── YOLO voitures ────────────────────────────────────
    resultats = modele(image_bgr, verbose=False)
    cibles = {
        "car": ("Voiture", (0, 200, 0), 1.5),
    }
    objets_detectes = []
    for r in resultats:
        for box in r.boxes:
            classe = modele.names[int(box.cls)]
            if classe in cibles:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label, couleur, hauteur_reelle = cibles[classe]
                pixels_hauteur = y2 - y1
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), couleur, 2)
                texte = f"{label} | {pixels_hauteur}px"
                cv2.putText(image_bgr, texte, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, couleur, 2)
                cx = (x1 + x2) // 2
                cv2.arrowedLine(image_bgr, (cx, y2), (cx, y1),
                               couleur, 1, tipLength=0.05)
                objets_detectes.append({
                    "label": label,
                    "pixels": pixels_hauteur,
                    "hauteur_reelle": hauteur_reelle,
                })

    # ── Ton modèle portes ────────────────────────────────
    try:
        modele_portes = charger_modele_portes()
        resultats_portes = modele_portes(image_bgr, verbose=False, conf=0.3)
        for r in resultats_portes:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                pixels_hauteur = y2 - y1
                couleur = (255, 0, 255)
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), couleur, 2)
                cv2.putText(image_bgr, f"Porte | {pixels_hauteur}px | {conf:.0%}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, couleur, 2)
                cx = (x1 + x2) // 2
                cv2.arrowedLine(image_bgr, (cx, y2), (cx, y1),
                               couleur, 1, tipLength=0.05)
                objets_detectes.append({
                    "label": "Porte",
                    "pixels": pixels_hauteur,
                    "hauteur_reelle": 2.2,
                })
    except Exception as e:
        print(f"Erreur modele portes : {e}")

    image_resultat = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_resultat), objets_detectes
# ── CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    .header-bar {
        background-color: #FFD700;
        padding: 8px 16px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .photo-card {
        background: white;
        border-radius: 8px;
        padding: 7px;
        margin-bottom: 8px;
        border: 2px solid #ddd;
        cursor: pointer;
        transition: border 0.2s;
    }
    .photo-card:hover { border-color: #FFD700; }
    .photo-card.selected { border-color: #e74c3c; }
    .meta-small { font-size: 11px; color: #555; margin-top: 4px; }
    .hauteur-badge {
        background: #27ae60;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
    }
    .section-title {
        font-size: 15px;
        font-weight: 700;
        color: #2c3e50;
        border-bottom: 2px solid #FFD700;
        padding-bottom: 3px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
    <span style="font-size:20px; font-weight:bold; color:#2c3e50;">
        🏗️ LNE — Analyse de Batiments &nbsp;&nbsp;
        <span style="background:#2c3e50;color:white;padding:4px 12px;border-radius:5px;font-size:14px;">
            Zendantennes
        </span>
    </span>
</div>
""", unsafe_allow_html=True)

col_f1, _ = st.columns([2, 5])
with col_f1:
    num_dossier = st.text_input("Numero de dossier", value="File 00000001")

modele = charger_modele()

photos = st.file_uploader(
    "Importe les photos",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if photos:
    # Collecter toutes les données
    photos_data = []
    for photo in photos:
        photo.seek(0); lat, lon = extraire_gps(photo)
        photo.seek(0); azimut   = extraire_azimut(photo)
        photo.seek(0); date     = extraire_date(photo)
        photos_data.append({"nom": photo.name, "lat": lat, "lon": lon,
                             "azimut": azimut, "date": date})

    # Initialiser photo sélectionnée
    if "photo_selectionnee" not in st.session_state:
        st.session_state.photo_selectionnee = 0

    # ── Layout : carte+hauteur gauche | grille photos droite
    col_gauche, col_droite = st.columns([5, 3])

    with col_droite:
        st.markdown('<div class="section-title">📷 Pictures</div>', unsafe_allow_html=True)

        # Grille 2 colonnes
        for row in range(0, len(photos), 2):
            cols = st.columns(2)
            for col_idx in range(2):
                idx = row + col_idx
                if idx >= len(photos):
                    break
                photo = photos[idx]
                d = photos_data[idx]
                photo.seek(0)
                image = Image.open(photo)
                image = corriger_rotation_auto(image, idx)
                image_mini = image.copy()
                image_mini.thumbnail((200, 130))

                with cols[col_idx]:
                    est_selec = st.session_state.photo_selectionnee == idx
                    border = "2px solid #e74c3c" if est_selec else "1px solid #ddd"

                    st.markdown(f'<div style="background:white;border-radius:8px;padding:6px;border:{border};margin-bottom:8px;">', unsafe_allow_html=True)
                    st.image(image_mini, use_container_width=True)
                    st.markdown(f"""
                    <div class="meta-small">
                        <b>Date</b> {d['date']}<br>
                        <b>Azimut</b> {f"{d['azimut']:.1f}°" if d['azimut'] else 'N/A'}<br>
                        <b>X</b> {f"{d['lon']:.2f}" if d['lon'] else 'N/A'}<br>
                        <b>Y</b> {f"{d['lat']:.2f}" if d['lat'] else 'N/A'}
                    </div>
                    """, unsafe_allow_html=True)

                    if f"hauteur_{idx}" in st.session_state:
                        st.markdown(f'<span class="hauteur-badge">⬆ {st.session_state[f"hauteur_{idx}"]:.2f} m</span>', unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                    if st.button(f"Voir", key=f"btn_{idx}"):
                        st.session_state.photo_selectionnee = idx
                        st.rerun()

    with col_gauche:
        idx_sel = st.session_state.photo_selectionnee
        d_sel   = photos_data[idx_sel]

        # Carte
        st.markdown(f'<div class="section-title">🗺️ Position — {photos[idx_sel].name}</div>', unsafe_allow_html=True)
        carte_html = afficher_carte_une_photo(d_sel)
        if carte_html:
            html(carte_html, height=380)
        else:
            st.warning("Pas de GPS pour cette photo")

        # ── Partie Hauteur ───────────────────────────────
        st.markdown('<div class="section-title">📐 Calcul de la hauteur</div>', unsafe_allow_html=True)

        photo_sel = photos[idx_sel]
        photo_sel.seek(0)
        image_sel = Image.open(photo_sel)
        largeur_orig, hauteur_orig = image_sel.size
        scale = min(600 / largeur_orig, 1.0)
        largeur_affich = int(largeur_orig * scale)
        hauteur_affich = int(hauteur_orig * scale)
        image_affich   = image_sel.resize((largeur_affich, hauteur_affich))

        # YOLO
        with st.spinner("Detection IA..."):
            photo_sel.seek(0)
            image_yolo = Image.open(photo_sel)
            image_annotee, objets = detecter_objets(image_yolo, modele)
        image_annotee_resized = image_annotee.resize((largeur_affich, hauteur_affich))
        st.image(image_annotee_resized, caption="Objets detectes", use_container_width=True)

        # Etape 1 — Tracer bâtiment
        st.markdown("**Etape 1 — Trace le batiment**")
        canvas_bat = st_canvas(
            fill_color="rgba(255,0,0,0.1)",
            stroke_width=3,
            stroke_color="#FF0000",
            background_image=image_affich,
            update_streamlit=True,
            width=largeur_affich,
            height=hauteur_affich,
            drawing_mode="rect",
            key=f"canvas_bat_{idx_sel}",
        )

        pixels_batiment = None
        if canvas_bat.json_data and canvas_bat.json_data["objects"]:
            dernier = canvas_bat.json_data["objects"][-1]
            pixels_batiment = abs(int(dernier["height"] / scale))
            st.success(f"Batiment : **{pixels_batiment} pixels**")

        # Etape 2 — Référence
        if pixels_batiment:
            st.markdown("**Etape 2 — Choisis la reference**")
            options = []
            for obj in objets:
                options.append({
                    "label": f"{obj['label']} (auto) — {obj['pixels']} px = {obj['hauteur_reelle']} m",
                    "pixels": obj["pixels"],
                    "hauteur_reelle": obj["hauteur_reelle"],
                    "type": "auto"
                })
            options.append({"label": "Porte (manuelle) = 2.2 m",   "pixels": None, "hauteur_reelle": 2.2, "type": "manuel"})
            options.append({"label": "Garage (manuel) = 3.0 m",    "pixels": None, "hauteur_reelle": 3.0, "type": "manuel"})
            options.append({"label": "Voiture (manuelle) = 1.5 m", "pixels": None, "hauteur_reelle": 1.5, "type": "manuel"})

            labels = [o["label"] for o in options]
            choix  = st.selectbox("Reference", labels, key=f"ref_{idx_sel}")
            ref    = options[labels.index(choix)]

            if ref["type"] == "manuel":
                couleurs_ref = {
                    "Porte (manuelle) = 2.2 m":   "#AA00FF",
                    "Garage (manuel) = 3.0 m":    "#FF8800",
                    "Voiture (manuelle) = 1.5 m": "#00AAFF",
                }
                st.markdown("**Trace la reference sur l'image**")
                canvas_ref = st_canvas(
                    fill_color="rgba(0,0,0,0.05)",
                    stroke_width=3,
                    stroke_color=couleurs_ref[ref["label"]],
                    background_image=image_affich,
                    update_streamlit=True,
                    width=largeur_affich,
                    height=hauteur_affich,
                    drawing_mode="rect",
                    key=f"canvas_ref_{idx_sel}_{choix}",
                )
                if canvas_ref.json_data and canvas_ref.json_data["objects"]:
                    dernier_ref = canvas_ref.json_data["objects"][-1]
                    ref["pixels"] = abs(int(dernier_ref["height"] / scale))
                    st.success(f"Reference : **{ref['pixels']} px** = {ref['hauteur_reelle']} m")

            # Etape 3 — Résultat
            if ref["pixels"] and pixels_batiment:
                hauteur_m = (pixels_batiment / ref["pixels"]) * ref["hauteur_reelle"]
                st.session_state[f"hauteur_{idx_sel}"] = hauteur_m
                st.success(f"""
                **Formule :** ({pixels_batiment} ÷ {ref['pixels']}) × {ref['hauteur_reelle']} m

                ### Hauteur estimee : {hauteur_m:.2f} m
                """)