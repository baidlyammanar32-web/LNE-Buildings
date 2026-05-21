import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import exifread
import os
import math
import cv2
import numpy as np
from ultralytics import YOLO
import tkintermapview
import tkinter.messagebox
import requests
import threading

# ── Modèle YOLO ─────────────────────────────────────────
modele_yolo = YOLO("yolov8n.pt")

def arrondir_azimut(azimut):
    return round(azimut / 10) * 10

def lire_azimut(chemin):
    try:
        with open(chemin, "rb") as f:
            tags = exifread.process_file(f, details=False)
            az = tags.get("GPS GPSImgDirection")
            if az:
                v = az.values[0]
                return float(v.num) / float(v.den)
    except:
        pass
    return None

def lire_gps(chemin):
    try:
        with open(chemin, "rb") as f:
            tags = exifread.process_file(f, details=False)
            lat     = tags.get("GPS GPSLatitude")
            lat_ref = tags.get("GPS GPSLatitudeRef")
            lon     = tags.get("GPS GPSLongitude")
            lon_ref = tags.get("GPS GPSLongitudeRef")
            if lat and lon:
                def conv(v):
                    d = float(v[0].num)/float(v[0].den)
                    m = float(v[1].num)/float(v[1].den)
                    s = float(v[2].num)/float(v[2].den)
                    return d + m/60 + s/3600
                la = conv(lat.values)
                lo = conv(lon.values)
                if str(lat_ref) != "N": la = -la
                if str(lon_ref) != "E": lo = -lo
                return la, lo
    except:
        pass
    return None, None

def detecter_voitures(chemin):
    try:
        image = Image.open(chemin).convert("RGB")
        image = ImageOps.exif_transpose(image)
        image_np  = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        resultats = modele_yolo(image_bgr, verbose=False)
        voitures = []
        for r in resultats:
            for box in r.boxes:
                classe = modele_yolo.names[int(box.cls)]
                if classe == "car":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    pixels = y2 - y1
                    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    cv2.putText(image_bgr, f"Voiture | {pixels}px",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
                    cx = (x1 + x2) // 2
                    cv2.arrowedLine(image_bgr, (cx, y2), (cx, y1),
                                   (0, 200, 0), 1, tipLength=0.05)
                    voitures.append(pixels)
        image_result = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_result), voitures
    except Exception as e:
        print(f"Erreur YOLO: {e}")
        return None, []

def get_batiments_osm(lat, lon, azimut, rayon=150):
    """Récupère les bâtiments OSM et les classe par angle par rapport à l'azimut."""
    try:
        query = f"""
        [out:json][timeout:15];
        way["building"](around:{rayon},{lat},{lon});
        out geom;
        """
        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data=query, timeout=20
        )
        data = resp.json()
        batiments = []
        for el in data.get("elements", []):
            if "geometry" not in el:
                continue
            coords = [(n["lat"], n["lon"]) for n in el["geometry"]]
            if len(coords) < 3:
                continue
            clat = sum(c[0] for c in coords) / len(coords)
            clon = sum(c[1] for c in coords) / len(coords)
            angle_bat = math.degrees(math.atan2(clon - lon, clat - lat)) % 360
            diff = abs(azimut - angle_bat) % 360
            if diff > 180:
                diff = 360 - diff
            # Couleur selon proximité avec l'azimut
            if diff < 25:
                couleur = "#e74c3c"   # rouge vif = dans la photo
                opacite = 0.6
            elif diff < 50:
                couleur = "#e67e22"   # orange
                opacite = 0.5
            else:
                couleur = "#3498db"   # bleu = hors champ
                opacite = 0.3
            batiments.append({
                "coords": coords,
                "couleur": couleur,
                "opacite": opacite,
                "diff": diff
            })
        batiments.sort(key=lambda x: x["diff"])
        return batiments
    except Exception as e:
        print(f"Erreur OSM: {e}")
        return []

# ════════════════════════════════════════════════════════
#  ICÔNE CAMÉRA — dessinée sur un Canvas Tkinter
# ════════════════════════════════════════════════════════
def creer_icone_camera(taille=32, couleur="#FFD700", bg="#2c3e50"):
    """Crée une icône caméra comme PhotoImage."""
    img = Image.new("RGBA", (taille, taille), (0, 0, 0, 0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    m = taille // 8
    # Corps caméra
    draw.rounded_rectangle(
        [m, m*2, taille-m, taille-m],
        radius=m, fill=couleur
    )
    # Objectif
    cx, cy = taille // 2, taille // 2 + m // 2
    r = taille // 4
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=bg)
    draw.ellipse([cx-r//2, cy-r//2, cx+r//2, cy+r//2], fill=couleur)
    # Flash
    draw.rectangle(
        [taille//2 + m, m, taille//2 + m*2, m*2],
        fill=couleur
    )
    return ImageTk.PhotoImage(img)


class AzimutTool:
    def __init__(self, root):
        self.root = root
        self.root.title("LNE — Azimut + Hauteur Auto")
        self.root.state("zoomed")
        self.root.configure(bg="#1a252f")

        self.photos         = []
        self.idx_actuel     = -1
        self.chemin_actuel  = None
        self.voitures_px    = []
        self.lat_actuel     = None
        self.lon_actuel     = None
        self.az_actuel      = None

        self.rect_start     = None
        self.rect_id        = None
        self.mode_dessin    = None
        self.px_bat         = None
        self.px_ref         = None

        self.image_yolo_pil = None
        self.rotation_angle = 0
        self.zoom_factor    = 1.0
        self.image_tk       = None

        self._scroll_job    = None
        self._last_event_x  = 0
        self._last_event_y  = 0

        self._marker_actuel = None
        self._path_actuel   = None
        self._cone_items    = []   # objets du cône sur la carte
        self._polygones     = []

        self._osm_thread    = None

        self.construire_interface()
        self.root.bind("<Left>",  lambda e: self.photo_precedente())
        self.root.bind("<Right>", lambda e: self.photo_suivante())

    # ══════════════════════════════════════════════════════
    #  INTERFACE
    # ══════════════════════════════════════════════════════
    def construire_interface(self):

        # ── HEADER ──────────────────────────────────────
        header = tk.Frame(self.root, bg="#FFD700", pady=5)
        header.pack(fill="x")
        tk.Label(header, text="LNE — Azimut + Hauteur Automatique",
                 bg="#FFD700", fg="#1a252f",
                 font=("Arial", 14, "bold")).pack(side="left", padx=12)
        tk.Button(header, text="📁 Choisir dossier",
                  command=self.choisir_dossier,
                  bg="#1a252f", fg="white",
                  font=("Arial", 10, "bold"),
                  padx=8, pady=2, relief="flat",
                  cursor="hand2").pack(side="right", padx=12)
        self.label_dossier = tk.Label(header, text="Aucun dossier",
                                       bg="#FFD700", fg="#1a252f",
                                       font=("Arial", 9))
        self.label_dossier.pack(side="right", padx=6)

        # ── CORPS PRINCIPAL (PanedWindow vertical) ───────
        self.paned = tk.PanedWindow(self.root, orient="vertical",
                                     bg="#1a252f", sashwidth=6,
                                     sashrelief="flat",
                                     sashpad=2)
        self.paned.pack(fill="both", expand=True, padx=4, pady=4)

        # ════════════════════════════════════════════════
        #  MOITIÉ HAUTE : liste + carte
        # ════════════════════════════════════════════════
        frame_haut = tk.Frame(self.paned, bg="#1a252f")
        self.paned.add(frame_haut, minsize=200)

        # Liste photos (gauche, largeur fixe)
        frame_liste = tk.Frame(frame_haut, bg="#2c3e50", width=260)
        frame_liste.pack(side="left", fill="y", padx=(0, 4))
        frame_liste.pack_propagate(False)

        # Titre + navigation
        nav = tk.Frame(frame_liste, bg="#2c3e50")
        nav.pack(fill="x", padx=6, pady=(6, 2))
        tk.Label(nav, text="Photos", bg="#2c3e50", fg="#FFD700",
                 font=("Arial", 10, "bold")).pack(side="left")
        tk.Button(nav, text="◀", command=self.photo_precedente,
                  bg="#1a252f", fg="white", font=("Arial", 9, "bold"),
                  relief="flat", cursor="hand2", padx=4).pack(side="right")
        tk.Button(nav, text="▶", command=self.photo_suivante,
                  bg="#1a252f", fg="white", font=("Arial", 9, "bold"),
                  relief="flat", cursor="hand2", padx=4).pack(side="right", padx=2)

        sc_l = tk.Scrollbar(frame_liste, orient="vertical")
        sc_l.pack(side="right", fill="y")
        self.liste = tk.Listbox(frame_liste,
                                yscrollcommand=sc_l.set,
                                bg="#1a252f", fg="white",
                                selectbackground="#3498db",
                                font=("Consolas", 9), relief="flat",
                                cursor="hand2", activestyle="none",
                                borderwidth=0)
        self.liste.pack(fill="both", expand=True, padx=4, pady=2)
        sc_l.config(command=self.liste.yview)
        self.liste.bind("<<ListboxSelect>>", self._on_liste_select)

        # Carte (droite, prend le reste)
        frame_carte_outer = tk.Frame(frame_haut, bg="#1a252f")
        frame_carte_outer.pack(side="left", fill="both", expand=True)

        # Status de la carte
        self.label_carte_status = tk.Label(frame_carte_outer, text="",
                                            bg="#1a252f", fg="#aaa",
                                            font=("Arial", 8, "italic"))
        self.label_carte_status.pack(anchor="w", padx=4)

        self.carte = tkintermapview.TkinterMapView(
            frame_carte_outer, corner_radius=0
        )
        self.carte.pack(fill="both", expand=True)
        self.carte.set_tile_server(
            "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png")
        self.carte.set_position(50.85, 4.35)
        self.carte.set_zoom(14)

        # ════════════════════════════════════════════════
        #  MOITIÉ BASSE : photo + paramètres
        # ════════════════════════════════════════════════
        frame_bas = tk.Frame(self.paned, bg="#1a252f")
        self.paned.add(frame_bas, minsize=200)

        # ── Barre outils ──────────────────────────────────
        barre = tk.Frame(frame_bas, bg="#2c3e50", pady=3)
        barre.pack(fill="x", pady=(0, 3))

        b = dict(font=("Arial", 9, "bold"), relief="flat",
                 cursor="hand2", padx=7, pady=2)

        # Rotation avec entrée
        tk.Label(barre, text="Rotation :", bg="#2c3e50", fg="#aaa",
                 font=("Arial", 9)).pack(side="left", padx=(8, 2))
        self.rotation_var = tk.StringVar(value="0")
        rot_menu = tk.OptionMenu(barre, self.rotation_var,
                                  "0", "90", "180", "270",
                                  command=self._appliquer_rotation_menu)
        rot_menu.config(bg="#1a252f", fg="white", activebackground="#3498db",
                        font=("Arial", 9), relief="flat",
                        highlightthickness=0, cursor="hand2")
        rot_menu["menu"].config(bg="#1a252f", fg="white",
                                 font=("Arial", 9))
        rot_menu.pack(side="left", padx=2)

        tk.Button(barre, text="↺ G", command=self.rotation_gauche,
                  bg="#16a085", fg="white", **b).pack(side="left", padx=2)
        tk.Button(barre, text="↻ D", command=self.rotation_droite,
                  bg="#16a085", fg="white", **b).pack(side="left", padx=2)

        tk.Frame(barre, bg="#555", width=1).pack(side="left", fill="y", padx=6)

        tk.Button(barre, text="🔴 Tracer Batiment",
                  command=lambda: self.activer_dessin("batiment"),
                  bg="#e74c3c", fg="white", **b).pack(side="left", padx=2)
        tk.Button(barre, text="🟣 Tracer Reference",
                  command=lambda: self.activer_dessin("reference"),
                  bg="#8e44ad", fg="white", **b).pack(side="left", padx=2)

        self.label_mode = tk.Label(barre, text="", bg="#2c3e50",
                                    fg="#FFD700",
                                    font=("Arial", 8, "italic"))
        self.label_mode.pack(side="left", padx=6)

        # Zoom info
        self.label_zoom = tk.Label(barre, text="Zoom: 100%",
                                    bg="#2c3e50", fg="#aaa",
                                    font=("Arial", 8))
        self.label_zoom.pack(side="right", padx=8)

        # ── Zone principale bas ───────────────────────────
        zone_bas = tk.Frame(frame_bas, bg="#1a252f")
        zone_bas.pack(fill="both", expand=True)

        # Photo (gauche)
        frame_photo = tk.Frame(zone_bas, bg="#0d1b2a")
        frame_photo.pack(side="left", fill="both", expand=True, padx=(0, 3))

        self.sb_v = tk.Scrollbar(frame_photo, orient="vertical")
        self.sb_v.pack(side="right", fill="y")
        self.sb_h = tk.Scrollbar(frame_photo, orient="horizontal")
        self.sb_h.pack(side="bottom", fill="x")

        self.canvas_photo = tk.Canvas(frame_photo, bg="#0d1b2a",
                                       cursor="crosshair",
                                       highlightthickness=0,
                                       xscrollcommand=self.sb_h.set,
                                       yscrollcommand=self.sb_v.set)
        self.canvas_photo.pack(fill="both", expand=True)
        self.sb_v.config(command=self.canvas_photo.yview)
        self.sb_h.config(command=self.canvas_photo.xview)

        self.canvas_photo.bind("<ButtonPress-1>",   self.debut_rectangle)
        self.canvas_photo.bind("<B1-Motion>",        self.dessiner_rectangle)
        self.canvas_photo.bind("<ButtonRelease-1>",  self.fin_rectangle)
        self.canvas_photo.bind("<MouseWheel>",
                               lambda e: self.canvas_photo.yview_scroll(
                                   int(-1*(e.delta/120)), "units"))
        self.canvas_photo.bind("<Control-MouseWheel>", self._zoom_image)

        # ── Panneau paramètres (droite, largeur fixe) ─────
        frame_params = tk.Frame(zone_bas, bg="#2c3e50", width=270)
        frame_params.pack(side="left", fill="y")
        frame_params.pack_propagate(False)

        # Scrollable
        sc_p = tk.Scrollbar(frame_params, orient="vertical")
        sc_p.pack(side="right", fill="y")
        cv_p = tk.Canvas(frame_params, bg="#2c3e50",
                          highlightthickness=0,
                          yscrollcommand=sc_p.set, width=250)
        cv_p.pack(side="left", fill="both", expand=True)
        sc_p.config(command=cv_p.yview)

        inner = tk.Frame(cv_p, bg="#2c3e50")
        cv_p.create_window((0, 0), window=inner, anchor="nw", width=250)
        inner.bind("<Configure>",
                   lambda e: cv_p.config(
                       scrollregion=cv_p.bbox("all")))

        def scroll_p(e):
            cv_p.yview_scroll(int(-1*(e.delta/120)), "units")
        cv_p.bind("<MouseWheel>", scroll_p)
        inner.bind("<MouseWheel>", scroll_p)

        def sec(txt, color="#FFD700"):
            f = tk.Frame(inner, bg="#1a252f", height=24)
            f.pack(fill="x", padx=6, pady=(10, 2))
            f.pack_propagate(False)
            tk.Label(f, text=txt, bg="#1a252f", fg=color,
                     font=("Arial", 9, "bold")).pack(
                         side="left", padx=6)
            f.bind("<MouseWheel>", scroll_p)

        # ── Azimut ────────────────────────────────────────
        sec("◈  Azimut")
        self.label_az_brut = tk.Label(inner, text="Brut : —",
                                       bg="#2c3e50", fg="#aaa",
                                       font=("Arial", 9))
        self.label_az_brut.pack(anchor="w", padx=10)

        self.label_az_arrondi = tk.Label(inner, text="—",
                                          bg="#FFD700", fg="#1a252f",
                                          font=("Arial", 32, "bold"),
                                          anchor="center")
        self.label_az_arrondi.pack(fill="x", padx=8, pady=3)

        # ── GPS ───────────────────────────────────────────
        sec("◈  GPS")
        self.label_gps = tk.Label(inner, text="—",
                                   bg="#2c3e50", fg="white",
                                   font=("Consolas", 9),
                                   justify="left")
        self.label_gps.pack(anchor="w", padx=10)

        # ── Calcul Hauteur ────────────────────────────────
        sec("◈  Calcul Hauteur")

        self.label_voitures = tk.Label(inner, text="Voitures : —",
                                        bg="#2c3e50", fg="#2ecc71",
                                        font=("Arial", 9))
        self.label_voitures.pack(anchor="w", padx=10, pady=(2, 0))

        self.label_px_bat = tk.Label(inner, text="Batiment : — px",
                                      bg="#2c3e50", fg="white",
                                      font=("Arial", 9))
        self.label_px_bat.pack(anchor="w", padx=10)

        # Référence
        fr_ref = tk.Frame(inner, bg="#2c3e50")
        fr_ref.pack(fill="x", padx=10, pady=(4, 0))
        tk.Label(fr_ref, text="Référence :", bg="#2c3e50",
                 fg="#aaa", font=("Arial", 9)).pack(anchor="w")

        self.ref_var = tk.StringVar(value="Voiture (1.5m)")
        for r in ["Voiture (1.5m)", "Porte (2.2m)", "Garage (3.0m)"]:
            rb = tk.Radiobutton(fr_ref, text=r,
                                variable=self.ref_var, value=r,
                                bg="#2c3e50", fg="white",
                                selectcolor="#1a252f",
                                activebackground="#2c3e50",
                                font=("Arial", 9),
                                command=self.calculer_hauteur)
            rb.pack(anchor="w", padx=8)
            rb.bind("<MouseWheel>", scroll_p)

        self.label_px_ref = tk.Label(inner, text="Reference : — px",
                                      bg="#2c3e50", fg="white",
                                      font=("Arial", 9))
        self.label_px_ref.pack(anchor="w", padx=10, pady=(4, 0))

        # Résultat hauteur
        tk.Frame(inner, bg="#444", height=1).pack(fill="x", padx=8, pady=6)
        tk.Label(inner, text="Hauteur estimée", bg="#2c3e50",
                 fg="#aaa", font=("Arial", 9)).pack()
        self.label_hauteur = tk.Label(inner, text="— m",
                                       bg="#27ae60", fg="white",
                                       font=("Arial", 26, "bold"),
                                       anchor="center")
        self.label_hauteur.pack(fill="x", padx=8, pady=(2, 8))

        # Bind scroll sur les labels
        for w in [self.label_az_brut, self.label_az_arrondi,
                  self.label_gps, self.label_voitures,
                  self.label_px_bat, self.label_px_ref,
                  self.label_hauteur]:
            w.bind("<MouseWheel>", scroll_p)

    # ══════════════════════════════════════════════════════
    #  NAVIGATION
    # ══════════════════════════════════════════════════════
    def _on_liste_select(self, event):
        sel = self.liste.curselection()
        if sel:
            self.charger_photo(sel[0])

    def photo_precedente(self):
        if self.idx_actuel > 0:
            self.charger_photo(self.idx_actuel - 1)

    def photo_suivante(self):
        if self.idx_actuel < len(self.photos) - 1:
            self.charger_photo(self.idx_actuel + 1)

    # ══════════════════════════════════════════════════════
    #  ZOOM IMAGE
    # ══════════════════════════════════════════════════════
    def _zoom_image(self, event):
        if self.image_yolo_pil is None:
            return
        if event.delta > 0:
            self.zoom_factor = min(self.zoom_factor * 1.15, 6.0)
        else:
            self.zoom_factor = max(self.zoom_factor / 1.15, 0.15)
        self.label_zoom.config(text=f"Zoom: {int(self.zoom_factor*100)}%")
        self._afficher_image_zoom()

    def _afficher_image_zoom(self):
        if self.image_yolo_pil is None:
            return
        img = self.image_yolo_pil.rotate(self.rotation_angle, expand=True)
        nw  = max(1, int(img.width  * self.zoom_factor))
        nh  = max(1, int(img.height * self.zoom_factor))
        img_r = img.resize((nw, nh), Image.LANCZOS)
        self.canvas_photo.delete("all")
        self.image_tk = ImageTk.PhotoImage(img_r)
        self.canvas_photo.create_image(0, 0, anchor="nw", image=self.image_tk)
        self.canvas_photo.config(scrollregion=(0, 0, nw, nh))

    def _afficher_image(self, pil_image):
        self.image_yolo_pil = pil_image
        self.zoom_factor    = 1.0
        self.rotation_angle = 0
        self.rotation_var.set("0")
        self.label_zoom.config(text="Zoom: 100%")
        self._afficher_image_zoom()

    # ══════════════════════════════════════════════════════
    #  ROTATION
    # ══════════════════════════════════════════════════════
    def _appliquer_rotation_menu(self, val):
        self.rotation_angle = int(val)
        self._afficher_image_zoom()
        self._sauver_rotation_auto()

    def rotation_gauche(self):
        if self.image_yolo_pil is None:
            return
        self.rotation_angle = (self.rotation_angle + 90) % 360
        self.rotation_var.set(str(self.rotation_angle))
        self._afficher_image_zoom()
        self._sauver_rotation_auto()

    def rotation_droite(self):
        if self.image_yolo_pil is None:
            return
        self.rotation_angle = (self.rotation_angle - 90) % 360
        self.rotation_var.set(str(self.rotation_angle))
        self._afficher_image_zoom()
        self._sauver_rotation_auto()

    def _sauver_rotation_auto(self):
        """Sauvegarde automatique de la rotation dans le fichier."""
        if self.image_yolo_pil is None or not self.chemin_actuel:
            return
        if self.rotation_angle == 0:
            return
        try:
            img_rot = self.image_yolo_pil.rotate(
                self.rotation_angle, expand=True)
            img_rot.save(self.chemin_actuel)
            self.image_yolo_pil = img_rot
            self.rotation_angle = 0
            self.rotation_var.set("0")
            self._afficher_image_zoom()
            self.label_mode.config(text="✓ Rotation sauvée")
            self.root.after(2000, lambda: self.label_mode.config(text=""))
        except Exception as e:
            print(f"Erreur sauvegarde rotation: {e}")

    # ══════════════════════════════════════════════════════
    #  DESSIN RECTANGLE
    # ══════════════════════════════════════════════════════
    def activer_dessin(self, mode):
        self.mode_dessin = mode
        self.label_mode.config(
            text="● Tracer Batiment..." if mode == "batiment"
            else "● Tracer Reference..."
        )

    def debut_rectangle(self, event):
        if not self.mode_dessin:
            return
        x = self.canvas_photo.canvasx(event.x)
        y = self.canvas_photo.canvasy(event.y)
        self.rect_start = (x, y)
        if self.rect_id:
            self.canvas_photo.delete(self.rect_id)
            self.rect_id = None
        self._last_event_x = event.x
        self._last_event_y = event.y
        self._auto_scroll()

    def dessiner_rectangle(self, event):
        if not self.rect_start or not self.mode_dessin:
            return
        self._last_event_x = event.x
        self._last_event_y = event.y
        x = self.canvas_photo.canvasx(event.x)
        y = self.canvas_photo.canvasy(event.y)
        couleur = "#FF3333" if self.mode_dessin == "batiment" else "#BB00FF"
        if self.rect_id:
            self.canvas_photo.delete(self.rect_id)
        self.rect_id = self.canvas_photo.create_rectangle(
            self.rect_start[0], self.rect_start[1], x, y,
            outline=couleur, width=2,
            dash=(6, 3) if self.mode_dessin == "reference" else None
        )

    def fin_rectangle(self, event):
        self._stop_auto_scroll()
        if not self.rect_start or not self.mode_dessin:
            return
        y = self.canvas_photo.canvasy(event.y)
        hauteur_px = int(abs(y - self.rect_start[1]) / self.zoom_factor)

        if self.mode_dessin == "batiment":
            self.px_bat = hauteur_px
            self.label_px_bat.config(text=f"Batiment : {hauteur_px} px")
        else:
            self.px_ref = hauteur_px
            self.label_px_ref.config(text=f"Reference : {hauteur_px} px")

        self.mode_dessin = None
        self.label_mode.config(text="")
        self.calculer_hauteur()

    def calculer_hauteur(self):
        try:
            if not self.px_bat or not self.px_ref:
                return
            refs  = {"Voiture (1.5m)": 1.5,
                     "Porte (2.2m)": 2.2,
                     "Garage (3.0m)": 3.0}
            h_ref = refs[self.ref_var.get()]
            h     = (self.px_bat / self.px_ref) * h_ref
            self.label_hauteur.config(text=f"{h:.2f} m")
        except:
            self.label_hauteur.config(text="— m")

    # ══════════════════════════════════════════════════════
    #  AUTO-SCROLL (lent)
    # ══════════════════════════════════════════════════════
    def _auto_scroll(self):
        if self.rect_start is None:
            return
        canvas = self.canvas_photo
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        x  = self._last_event_x
        y  = self._last_event_y
        marge = 55
        scrolled = False

        if x < marge:
            canvas.xview_scroll(-1, "units"); scrolled = True
        elif x > cw - marge:
            canvas.xview_scroll(1, "units");  scrolled = True
        if y < marge:
            canvas.yview_scroll(-1, "units"); scrolled = True
        elif y > ch - marge:
            canvas.yview_scroll(1, "units");  scrolled = True

        if scrolled and self.rect_id and self.mode_dessin:
            cx = canvas.canvasx(x)
            cy = canvas.canvasy(y)
            couleur = "#FF3333" if self.mode_dessin == "batiment" else "#BB00FF"
            canvas.delete(self.rect_id)
            self.rect_id = canvas.create_rectangle(
                self.rect_start[0], self.rect_start[1], cx, cy,
                outline=couleur, width=2)

        self._scroll_job = self.root.after(70, self._auto_scroll)

    def _stop_auto_scroll(self):
        if self._scroll_job:
            self.root.after_cancel(self._scroll_job)
            self._scroll_job = None

    # ══════════════════════════════════════════════════════
    #  CARTE — marqueur caméra + cône azimut + cadastres
    # ══════════════════════════════════════════════════════
    def _effacer_carte(self):
        if self._marker_actuel:
            try: self._marker_actuel.delete()
            except: pass
            self._marker_actuel = None
        if self._path_actuel:
            try: self._path_actuel.delete()
            except: pass
            self._path_actuel = None
        for item in self._cone_items:
            try: item.delete()
            except: pass
        self._cone_items = []
        for p in self._polygones:
            try: p.delete()
            except: pass
        self._polygones = []

    def _dessiner_cone_azimut(self, lat, lon, azimut, distance=0.0008):
        """
        Dessine un cône qui représente le champ de vision de la caméra
        (±35° autour de l'azimut).
        """
        ouverture = 35  # degrés de chaque côté

        az_g = math.radians(azimut - ouverture)
        az_d = math.radians(azimut + ouverture)
        az_c = math.radians(azimut)

        # Points du cône
        lat_g = lat + distance * math.cos(az_g)
        lon_g = lon + distance * math.sin(az_g)
        lat_d = lat + distance * math.cos(az_d)
        lon_d = lon + distance * math.sin(az_d)
        lat_c = lat + distance * math.cos(az_c)
        lon_c = lon + distance * math.sin(az_c)

        # Polygone du cône (rempli, semi-transparent)
        try:
            cone = self.carte.set_polygon(
                [(lat, lon), (lat_g, lon_g),
                 (lat_c, lon_c), (lat_d, lon_d), (lat, lon)],
                fill_color="#FFD700",
                outline_color="#FF8C00",
                border_width=2,
                fill_opacity=0.25,
                name="cone_azimut"
            )
            self._cone_items.append(cone)
        except:
            # Si set_polygon non supporté, utiliser un chemin
            ligne = self.carte.set_path(
                [(lat_g, lon_g), (lat, lon), (lat_d, lon_d)],
                color="#FFD700", width=2
            )
            self._cone_items.append(ligne)

        # Ligne centrale (direction exacte)
        try:
            ligne_c = self.carte.set_path(
                [(lat, lon), (lat_c, lon_c)],
                color="#FF4500", width=2
            )
            self._cone_items.append(ligne_c)
        except:
            pass

    def _charger_batiments_thread(self, lat, lon, azimut):
        """Lance OSM dans un thread séparé."""
        def _run():
            batiments = get_batiments_osm(lat, lon, azimut)
            self.root.after(0, lambda: self._dessiner_batiments(batiments))
        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def _dessiner_batiments(self, batiments):
        for p in self._polygones:
            try: p.delete()
            except: pass
        self._polygones = []

        for bat in batiments:
            try:
                p = self.carte.set_polygon(
                    bat["coords"],
                    fill_color=bat["couleur"],
                    outline_color=bat["couleur"],
                    border_width=2,
                    fill_opacity=bat["opacite"],
                )
                if p:
                    self._polygones.append(p)
            except:
                pass

        nb = len(self._polygones)
        rouge = sum(1 for b in batiments if b["couleur"] == "#e74c3c")
        self.label_carte_status.config(
            text=f"{nb} bâtiments  |  {rouge} dans le champ de vision"
        )

    # ══════════════════════════════════════════════════════
    #  CHARGEMENT
    # ══════════════════════════════════════════════════════
    def choisir_dossier(self):
        dossier = filedialog.askdirectory(title="Choisir dossier")
        if not dossier:
            return
        self.label_dossier.config(text=os.path.basename(dossier))
        self.liste.delete(0, tk.END)
        self.photos = []
        for f in sorted(os.listdir(dossier)):
            if os.path.splitext(f)[1].lower() in [".jpg", ".jpeg", ".png"]:
                self.photos.append(os.path.join(dossier, f))
                self.liste.insert(tk.END, f"  {f}")

    def charger_photo(self, idx):
        if idx < 0 or idx >= len(self.photos):
            return
        self.idx_actuel    = idx
        self.chemin_actuel = self.photos[idx]

        self.liste.selection_clear(0, tk.END)
        self.liste.selection_set(idx)
        self.liste.see(idx)

        # Reset
        self.px_bat = None
        self.px_ref = None
        self.mode_dessin = None
        self.rect_start  = None
        self.rect_id     = None
        self._stop_auto_scroll()
        self.label_px_bat.config(text="Batiment : — px")
        self.label_px_ref.config(text="Reference : — px")
        self.label_hauteur.config(text="— m")
        self.label_mode.config(text="")
        self.label_carte_status.config(text="Chargement bâtiments...")

        # Effacer carte
        self._effacer_carte()

        # YOLO
        image_yolo, voitures = detecter_voitures(self.chemin_actuel)
        self.voitures_px = voitures
        if image_yolo:
            self.image_orig_pil = ImageOps.exif_transpose(image_yolo)
            self._afficher_image(self.image_orig_pil)

        if voitures:
            self.label_voitures.config(
                text=f"{len(voitures)} voiture(s) : " +
                     ", ".join([f"{px}px" for px in voitures])
            )
            self.px_ref = voitures[0]
            self.label_px_ref.config(
                text=f"Reference : {voitures[0]} px (auto)")
        else:
            self.label_voitures.config(text="Aucune voiture détectée")

        # Azimut
        az = lire_azimut(self.chemin_actuel)
        self.az_actuel = az
        if az is not None:
            az_arr = arrondir_azimut(az)
            self.label_az_brut.config(text=f"Brut : {az:.1f}°")
            self.label_az_arrondi.config(text=f"{az_arr}°")
        else:
            self.label_az_brut.config(text="Brut : —")
            self.label_az_arrondi.config(text="—")

        # GPS + carte
        lat, lon = lire_gps(self.chemin_actuel)
        self.lat_actuel = lat
        self.lon_actuel = lon

        if lat and lon:
            self.label_gps.config(
                text=f"Lat : {lat:.6f}\nLon : {lon:.6f}")
            self.carte.set_position(lat, lon)
            self.carte.set_zoom(18)

            # Marqueur caméra
            nom = os.path.basename(self.chemin_actuel)
            self._marker_actuel = self.carte.set_marker(
                lat, lon,
                text=f"📷 {nom}",
                marker_color_circle="#FFD700",
                marker_color_outside="#FF8C00"
            )

            if az is not None:
                az_arr = arrondir_azimut(az)
                # Cône de vision au lieu d'une ligne
                self._dessiner_cone_azimut(lat, lon, az_arr)
                # Bâtiments OSM en arrière-plan
                self._charger_batiments_thread(lat, lon, az_arr)
            else:
                self.label_carte_status.config(text="Pas d'azimut")
        else:
            self.label_gps.config(text="GPS : non disponible")
            self.label_carte_status.config(text="Pas de GPS")


root = tk.Tk()
app  = AzimutTool(root)
root.mainloop()