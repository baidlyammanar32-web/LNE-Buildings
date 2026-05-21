import streamlit as st
from PIL import Image, ImageOps
import exifread
import folium
import math
import cv2
import numpy as np
from streamlit.components.v1 import html
from ultralytics import YOLO
from streamlit_drawable_canvas import st_canvas
from streamlit_paste_button import paste_image_button
from datetime import datetime
import os
import io
import base64

# ════════════════════════════════════════════════════════
#  MODÈLES
# ════════════════════════════════════════════════════════
@st.cache_resource
def charger_modele_yolo():
    return YOLO("yolov8n.pt")

@st.cache_resource
def charger_modele_portes():
    path = r"C:\Users\pc\Desktop\RCP_simulator\nn\runs\detect\modele_portes\weights\best.pt"
    if not os.path.exists(path):
        return None
    return YOLO(path)

# ════════════════════════════════════════════════════════
#  UTILITAIRES IMAGE
# ════════════════════════════════════════════════════════
def pil_to_base64(image_pil):
    buf = io.BytesIO()
    image_pil.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

def base64_to_pil(data_url):
    header, encoded = data_url.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")

# ════════════════════════════════════════════════════════
#  IMAGE ZOOMABLE
# ════════════════════════════════════════════════════════
def image_zoomable(image_pil, height=500, caption=""):
    buf = io.BytesIO()
    image_pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    data_url = f"data:image/png;base64,{b64}"
    cap_html = f"<div style='font-size:11px;color:#8fa8bf;text-align:center;margin-top:4px;letter-spacing:.5px;'>{caption}</div>" if caption else ""
    html_code = f"""
    <div style="position:relative;width:100%;height:{height}px;
                background:linear-gradient(160deg,#1b2d3e 0%,#0f1e2d 100%);
                border-radius:6px;overflow:hidden;
                border:1px solid #2a4a62;">
      <canvas id="zc_{hash(data_url)%99999}" style="width:100%;height:100%;cursor:grab;display:block;"></canvas>
      <div style="position:absolute;top:6px;right:8px;font-size:10px;
                  color:#e8c840;background:rgba(10,22,35,0.85);
                  padding:2px 8px;border-radius:4px;pointer-events:none;letter-spacing:.4px;">
        Molette = zoom &nbsp;|&nbsp; Drag = déplacer
      </div>
      <button onclick="rv_{hash(data_url)%99999}()" style="position:absolute;bottom:8px;right:8px;
        background:#e8c840;color:#0f1e2d;border:none;border-radius:4px;
        font-size:11px;font-weight:700;padding:4px 10px;cursor:pointer;letter-spacing:.5px;">↺ Reset</button>
    </div>
    {cap_html}
    <script>
    (function() {{
      var cid    = 'zc_{hash(data_url)%99999}';
      var canvas = document.getElementById(cid);
      var ctx    = canvas.getContext('2d');
      var img    = new Image();
      var scale=1, minS=0.2, maxS=8, ox=0, oy=0;
      var drag=false, lx=0, ly=0;
      function resize(){{ canvas.width=canvas.offsetWidth; canvas.height=canvas.offsetHeight; draw(); }}
      function fit(){{ var sw=canvas.width/img.width, sh=canvas.height/img.height;
        scale=Math.min(sw,sh); ox=(canvas.width-img.width*scale)/2; oy=(canvas.height-img.height*scale)/2; }}
      function draw(){{ ctx.clearRect(0,0,canvas.width,canvas.height);
        ctx.save(); ctx.translate(ox,oy); ctx.scale(scale,scale); ctx.drawImage(img,0,0); ctx.restore(); }}
      img.onload=function(){{ resize(); fit(); draw(); }};
      img.src='{data_url}';
      canvas.addEventListener('wheel',function(e){{
        e.preventDefault();
        var r=canvas.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
        var d=e.deltaY<0?1.15:0.87, ns=Math.min(maxS,Math.max(minS,scale*d));
        ox=mx-(mx-ox)*(ns/scale); oy=my-(my-oy)*(ns/scale); scale=ns; draw();
      }},{{passive:false}});
      canvas.addEventListener('mousedown',function(e){{ drag=true; lx=e.clientX; ly=e.clientY; canvas.style.cursor='grabbing'; }});
      window.addEventListener('mousemove',function(e){{ if(!drag)return; ox+=e.clientX-lx; oy+=e.clientY-ly; lx=e.clientX; ly=e.clientY; draw(); }});
      window.addEventListener('mouseup',function(){{ drag=false; canvas.style.cursor='grab'; }});
      window['rv_{hash(data_url)%99999}']=function(){{ fit(); draw(); }};
      window.addEventListener('resize',function(){{ resize(); fit(); draw(); }});
    }})();
    </script>
    """
    st.components.v1.html(html_code, height=height+30)

# ════════════════════════════════════════════════════════
#  ROTATION AUTO
# ════════════════════════════════════════════════════════
def detecter_ciel_et_corriger(image_pil):
    try:
        scores = {}
        for angle in [0, 90, 180, 270]:
            img  = np.array(image_pil.rotate(angle, expand=True).resize((224,224)))
            haut = img[:56,  :, :].astype(float)
            bas  = img[168:, :, :].astype(float)
            def score(z):
                r,g,b = z[:,:,0],z[:,:,1],z[:,:,2]
                return (b-r).mean() + z.mean()*0.3
            scores[angle] = score(haut) - score(bas)
        return max(scores, key=scores.get)
    except:
        return 0

def corriger_rotation_auto(image_pil, idx):
    angle_manuel = st.session_state.get(f"rotation_{idx}", None)
    if angle_manuel is not None:
        return image_pil.rotate(angle_manuel, expand=True)
    angle = detecter_ciel_et_corriger(image_pil)
    st.session_state[f"angle_auto_{idx}"] = angle
    if angle != 0:
        return image_pil.rotate(angle, expand=True)
    return image_pil

# ════════════════════════════════════════════════════════
#  EXIF
# ════════════════════════════════════════════════════════
def extraire_gps(photo):
    tags = exifread.process_file(photo, details=False)
    try:
        lat=tags["GPS GPSLatitude"].values; lat_ref=tags["GPS GPSLatitudeRef"].values
        lon=tags["GPS GPSLongitude"].values; lon_ref=tags["GPS GPSLongitudeRef"].values
        def conv(v):
            return float(v[0].num)/float(v[0].den)+float(v[1].num)/float(v[1].den)/60+float(v[2].num)/float(v[2].den)/3600
        la=conv(lat); lo=conv(lon)
        if lat_ref!="N": la=-la
        if lon_ref!="E": lo=-lo
        return la,lo
    except: return None,None

def extraire_azimut(photo):
    tags = exifread.process_file(photo, details=False)
    try:
        az=tags["GPS GPSImgDirection"].values[0]
        return float(az.num)/float(az.den)
    except: return None

def extraire_date(photo):
    tags = exifread.process_file(photo, details=False)
    try:
        return datetime.strptime(str(tags["EXIF DateTimeOriginal"]),"%Y:%m:%d %H:%M:%S").strftime("%d-%b-%Y")
    except: return datetime.now().strftime("%d-%b-%Y")

# ════════════════════════════════════════════════════════
#  CARTE
# ════════════════════════════════════════════════════════
def afficher_carte(d):
    if not d["lat"] or not d["lon"]: return None
    lat=d["lat"]; lon=d["lon"]; azimut=d["azimut"] if d["azimut"] else 0
    distance=0.0008; ouv=35
    def pt(az_deg):
        r=math.radians(az_deg)
        return [lat+distance*math.cos(r), lon+distance*math.sin(r)]
    pt_g=pt(azimut-ouv); pt_c=pt(azimut); pt_d=pt(azimut+ouv)
    carte=folium.Map(location=[lat,lon],zoom_start=18,tiles="OpenStreetMap")
    folium.Marker([lat,lon],
        icon=folium.DivIcon(
            html='<div style="width:32px;height:32px;background:#1b3a52;border-radius:50%;border:2px solid #e8c840;display:flex;align-items:center;justify-content:center;font-size:15px;">📷</div>',
            icon_size=(32,32),icon_anchor=(16,16)),
        popup=folium.Popup(f"<b>Photo</b><br>Azimut:{azimut:.0f}°<br>Lat:{lat:.6f}<br>Lon:{lon:.6f}",max_width=200)
    ).add_to(carte)
    folium.Polygon(locations=[[lat,lon],pt_g,pt_c,pt_d,[lat,lon]],
        color="#e8c840",fill=True,fill_color="#e8c840",fill_opacity=0.2,weight=2,
        tooltip=f"Direction — {azimut:.0f}°").add_to(carte)
    folium.PolyLine(locations=[[lat,lon],pt_c],color="#e05a2b",weight=2,dash_array="6 3").add_to(carte)
    osm_srcdoc=carte._repr_html_().replace("&","&amp;").replace('"',"&quot;")
    gsv_src=f"https://maps.google.com/maps?q={lat},{lon}&layer=c&cbll={lat},{lon}&cbp=12,{azimut:.0f},0,0,0&output=svembed"
    gsv_lien=f"https://www.google.com/maps/@{lat},{lon},3a,80y,{azimut:.0f}h,0t/data=!3m6!1e1"

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0f1e2d;font-family:'Segoe UI',Arial,sans-serif}}
  .panel{{width:100%;position:relative;overflow:hidden;border-radius:6px;}}
  .panel-sv{{height:420px;margin-bottom:8px;background:#0a151f;border:1px solid #2a4a62;}}
  .panel-osm{{height:380px;background:#0a151f;border:1px solid #2a4a62;}}
  .badge{{position:absolute;top:8px;left:8px;z-index:9999;background:rgba(10,21,31,0.9);color:#e8c840;font-size:11px;font-weight:700;padding:3px 10px;border-radius:3px;pointer-events:none;letter-spacing:.6px;text-transform:uppercase;border-left:2px solid #e8c840;}}
  iframe{{width:100%;height:100%;border:none}}
  .open-btn{{position:absolute;bottom:10px;right:10px;z-index:9999;background:#e8c840;color:#0f1e2d;font-size:11px;font-weight:700;padding:5px 12px;border-radius:3px;text-decoration:none;letter-spacing:.5px;}}
</style></head><body>
<div class="panel panel-sv">
  <div class="badge">🚶 Street View — {azimut:.0f}°</div>
  <iframe src="{gsv_src}" allowfullscreen loading="lazy"></iframe>
  <a class="open-btn" href="{gsv_lien}" target="_blank">↗ Ouvrir GSV</a>
</div>
<div class="panel panel-osm">
  <div class="badge">🗺️ Carte OSM</div>
  <iframe srcdoc="{osm_srcdoc}"></iframe>
</div>
</body></html>"""

# ════════════════════════════════════════════════════════
#  DÉTECTION OBJETS
# ════════════════════════════════════════════════════════
def detecter_objets(image_pil):
    modele=charger_modele_yolo()
    img_np=np.array(image_pil)
    img_bgr=cv2.cvtColor(img_np,cv2.COLOR_RGB2BGR)
    res=modele(img_bgr,verbose=False)
    objets=[]
    for r in res:
        for box in r.boxes:
            cls=modele.names[int(box.cls)]
            if cls=="car":
                x1,y1,x2,y2=map(int,box.xyxy[0]); px=y2-y1
                cv2.rectangle(img_bgr,(x1,y1),(x2,y2),(0,180,0),2)
                cv2.putText(img_bgr,f"Voiture|{px}px",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,180,0),2)
                cv2.arrowedLine(img_bgr,((x1+x2)//2,y2),((x1+x2)//2,y1),(0,180,0),1,tipLength=0.05)
                objets.append({"label":"Voiture","pixels":px,"hauteur_reelle":1.5})
    modele_p=charger_modele_portes()
    if modele_p:
        try:
            res_p=modele_p(img_bgr,verbose=False,conf=0.3)
            for r in res_p:
                for box in r.boxes:
                    x1,y1,x2,y2=map(int,box.xyxy[0]); conf=float(box.conf[0]); px=y2-y1
                    cv2.rectangle(img_bgr,(x1,y1),(x2,y2),(200,0,200),2)
                    cv2.putText(img_bgr,f"Porte|{px}px|{conf:.0%}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,0,200),2)
                    cv2.arrowedLine(img_bgr,((x1+x2)//2,y2),((x1+x2)//2,y1),(200,0,200),1,tipLength=0.05)
                    objets.append({"label":"Porte","pixels":px,"hauteur_reelle":2.2})
        except: pass
    return Image.fromarray(cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)),objets

# ════════════════════════════════════════════════════════
#  CONFIG & THÈME
# ════════════════════════════════════════════════════════
st.set_page_config(page_title="LNE — Analyse Bâtiments",layout="wide",initial_sidebar_state="collapsed")
st.markdown("""<style>
  .stApp, section[data-testid="stMain"] {
    background: linear-gradient(160deg, #e8edf2 0%, #d4dde6 40%, #c8d4de 100%) !important;
    background-attachment: fixed !important;
  }
  .lne-header {
    background: linear-gradient(90deg, #e8c840 0%, #c9a820 100%);
    padding: 10px 18px; border-radius: 5px; margin-bottom: 12px;
    display: flex; align-items: center; gap: 12px;
  }
  .lne-header span { font-size:19px; font-weight:700; color:#0f1e2d; letter-spacing:.3px; }
  .lne-tag { background:#0f1e2d;color:#e8c840;padding:3px 10px;border-radius:3px;font-size:12px;font-weight:700;letter-spacing:.8px; }
  .sec-title { font-size:11px;font-weight:700;color:#e8c840;border-bottom:2px solid #e8c840;padding-bottom:4px;margin-bottom:10px;text-transform:uppercase;letter-spacing:1.2px; }
  .meta { font-size:10px;color:#7a9bb5;margin-top:4px;font-family:monospace; }
  .meta b { color:#e8c840; }
  .hauteur-badge { display:inline-block;background:linear-gradient(90deg,#1b4a2e,#1d6636);color:#7de8a0;padding:2px 8px;border-radius:3px;font-size:11px;font-weight:700;margin-top:4px;border:1px solid #2a7a46; }
  .hauteur-result { background:linear-gradient(135deg,#1b4a2e 0%,#0f2d1a 100%);color:#7de8a0;border-radius:5px;padding:14px;text-align:center;font-size:30px;font-weight:700;margin-top:8px;border:1px solid #2a7a46; }
  .azimut-badge { background:linear-gradient(135deg,#2a3a0f 0%,#1a2808 100%);color:#e8c840;border-radius:5px;padding:14px;text-align:center;font-size:30px;font-weight:700;border:1px solid #4a5c1a;margin-top:8px; }
  .source-badge { display:inline-block;background:#2a1a4a;color:#b88de8;padding:3px 10px;border-radius:3px;font-size:11px;font-weight:700;margin-bottom:6px;border:1px solid #4a2a7a; }
  div.stButton > button { background:linear-gradient(135deg,#1b3a52 0%,#0f2538 100%) !important;border:1px solid #2a5a7a !important;color:#8fa8bf !important;border-radius:4px !important;font-weight:600 !important; }
  div.stButton > button:hover { border-color:#e8c840 !important;color:#e8c840 !important; }
  div[data-testid="stAlert"] { background:#162538 !important;border-left:3px solid #e8c840 !important;color:#c8dce8 !important;border-radius:4px !important; }
  ::-webkit-scrollbar { width:5px;height:5px; }
  ::-webkit-scrollbar-track { background:#0f1e2d; }
  ::-webkit-scrollbar-thumb { background:#e8c840;border-radius:3px; }
  #MainMenu,footer,header { visibility:hidden; }
  .block-container { padding-top:0.6rem !important; }
  details { background:#162538 !important;border:1px solid #2a4a62 !important;border-radius:4px !important; }
  summary { color:#8fa8bf !important; }
</style>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════
st.markdown("""<div class="lne-header">
  <span>🏗️ LNE — Analyse de Bâtiments &nbsp;
    <span class="lne-tag">ZENDANTENNES</span>
  </span>
</div>""", unsafe_allow_html=True)

col_f1, _ = st.columns([2, 6])
with col_f1:
    st.text_input("", value="File 00000001", placeholder="Numéro de dossier", label_visibility="collapsed")

# ════════════════════════════════════════════════════════
#  UPLOAD
# ════════════════════════════════════════════════════════
photos = st.file_uploader("Photos", type=["jpg","jpeg","png"], accept_multiple_files=True, label_visibility="collapsed")
if not photos:
    st.markdown(
        '<div style="text-align:center;padding:60px;color:#4a6a82;">'
        '<div style="font-size:48px;">📁</div>'
        '<div style="font-size:15px;margin-top:14px;letter-spacing:.5px;">Importe tes photos pour commencer</div>'
        '</div>', unsafe_allow_html=True)
    st.stop()

# ════════════════════════════════════════════════════════
#  EXIF
# ════════════════════════════════════════════════════════
photos_data = []
for photo in photos:
    photo.seek(0); lat, lon = extraire_gps(photo)
    photo.seek(0); azimut = extraire_azimut(photo)
    photo.seek(0); date = extraire_date(photo)
    az_arr = round(azimut / 10) * 10 if azimut else None
    photos_data.append({"nom": photo.name, "lat": lat, "lon": lon, "azimut": azimut, "az_arr": az_arr, "date": date})

if "idx_sel" not in st.session_state:
    st.session_state.idx_sel = 0

col_g, col_d = st.columns([6, 2], gap="small")

# ════════════════════════════════════════════════════════
#  DROITE — grille photos
# ════════════════════════════════════════════════════════
with col_d:
    st.markdown('<div class="sec-title">📷 Photos</div>', unsafe_allow_html=True)
    n1, n2, n3 = st.columns([1, 2, 1])
    with n1:
        if st.button("◀", use_container_width=True):
            if st.session_state.idx_sel > 0:
                st.session_state.idx_sel -= 1; st.rerun()
    with n2:
        st.markdown(
            f"<div style='text-align:center;color:#4a6a82;font-size:11px;padding-top:6px;'>"
            f"{st.session_state.idx_sel+1} / {len(photos)}</div>", unsafe_allow_html=True)
    with n3:
        if st.button("▶", use_container_width=True):
            if st.session_state.idx_sel < len(photos) - 1:
                st.session_state.idx_sel += 1; st.rerun()

    for idx in range(len(photos)):
        d = photos_data[idx]
        photo = photos[idx]; photo.seek(0)
        img = Image.open(photo)
        img = corriger_rotation_auto(img, idx)
        img.thumbnail((180, 130))
        est_sel = st.session_state.idx_sel == idx
        border = "2px solid #e8c840" if est_sel else "1px solid #2a4a62"
        bg = "linear-gradient(160deg,#243d52 0%,#1a2f42 100%)" if est_sel else "linear-gradient(160deg,#1e3348 0%,#162538 100%)"
        st.markdown(f'<div style="background:{bg};border-radius:6px;padding:6px;border:{border};margin-bottom:8px;">', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        az_txt = f"{d['az_arr']}°" if d['az_arr'] else "—"
        lon_txt = f"{d['lon']:.2f}" if d['lon'] else "N/A"
        lat_txt = f"{d['lat']:.2f}" if d['lat'] else "N/A"
        st.markdown(f'<div class="meta"><b>Date</b> {d["date"]}<br><b>Az</b> {az_txt} &nbsp;<b>X</b> {lon_txt} &nbsp;<b>Y</b> {lat_txt}</div>', unsafe_allow_html=True)
        if f"hauteur_{idx}" in st.session_state:
            st.markdown(f'<span class="hauteur-badge">⬆ {st.session_state[f"hauteur_{idx}"]:.2f} m</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("Voir", key=f"btn_{idx}", use_container_width=True):
            st.session_state.idx_sel = idx; st.rerun()

# ════════════════════════════════════════════════════════
#  GAUCHE — Cartes + Capture + Calcul
# ════════════════════════════════════════════════════════
with col_g:
    idx = st.session_state.idx_sel
    d = photos_data[idx]
    cap_key = f"capture_data_{idx}"

    az_disp = f"{d['az_arr']}°" if d['az_arr'] else "—"
    st.markdown(f'<div class="sec-title">🗺️ {d["nom"]} &nbsp;·&nbsp; Azimut {az_disp}</div>', unsafe_allow_html=True)

    carte_html = afficher_carte(d)
    if carte_html:
        html(carte_html, height=830)
    else:
        st.warning("Pas de données GPS pour cette photo")

    # ════════════════════════════════════════
    #  CAPTURE
    # ════════════════════════════════════════
    st.markdown('<div class="sec-title">📸 Capture d\'écran</div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:11px;color:#4a6a82;margin-bottom:8px;letter-spacing:.3px;'>"
        "1️⃣ PrintScreen &nbsp;·&nbsp; 2️⃣ Clique le bouton &nbsp;·&nbsp; 3️⃣ L'image s'affiche"
        "</div>", unsafe_allow_html=True)

    # ── CORRECTION : key inclut paste_ver pour forcer reset du composant ──
    paste_result = paste_image_button(
        label="📋  Coller depuis le presse-papiers (Ctrl+V)",
        key=f"paste_{idx}_{st.session_state.get(f'paste_ver_{idx}', 0)}",
        background_color="#162538",
        hover_background_color="#1e3348",
        text_color="#e8c840",
    )
    if paste_result.image_data is not None:
        paste_hash = hash(paste_result.image_data.tobytes())
        last_hash = st.session_state.get(f"paste_hash_{idx}", None)
        if paste_hash != last_hash:
            st.session_state[f"paste_hash_{idx}"] = paste_hash
            img_pasted = paste_result.image_data.convert("RGB")
            st.session_state[cap_key] = pil_to_base64(img_pasted)
            st.success("✅ Capture collée avec succès !")

    st.markdown("<div style='font-size:11px;color:#4a6a82;margin-top:6px;margin-bottom:2px;'>Ou importer un fichier :</div>", unsafe_allow_html=True)
    upload_cap = st.file_uploader("Importer une capture", type=["jpg","jpeg","png"], key=f"upload_{idx}", label_visibility="collapsed")
    if upload_cap:
        img_up = Image.open(upload_cap).convert("RGB")
        st.session_state[cap_key] = pil_to_base64(img_up)
        st.success("✅ Capture importée !")

    col_eff1, col_eff2 = st.columns([3, 1])
    with col_eff2:
        if st.session_state.get(cap_key):
            if st.button("🗑️ Effacer", key=f"eff_{idx}", use_container_width=True):
                st.session_state.pop(cap_key, None)
                # ── CORRECTION : incrémenter paste_ver pour détruire le composant ──
                st.session_state[f"paste_ver_{idx}"] = st.session_state.get(f"paste_ver_{idx}", 0) + 1
                st.session_state[f"paste_hash_{idx}"] = None
                st.session_state[f"canvas_ver_{idx}"] = st.session_state.get(f"canvas_ver_{idx}", 0) + 1
                st.rerun()

    has_capture = bool(st.session_state.get(cap_key, ""))
    if has_capture:
        img_prev = base64_to_pil(st.session_state[cap_key])
        image_zoomable(img_prev, height=250, caption="📋 Capture active")

    # ════════════════════════════════════════
    #  CALCUL HAUTEUR
    # ════════════════════════════════════════
    st.markdown('<div class="sec-title">📐 Calcul Hauteur</div>', unsafe_allow_html=True)

    source_opts = ["🖼️ Photo originale"]
    if has_capture:
        source_opts.insert(0, "📋 Capture active")
    source_sel = st.radio("Source :", source_opts, horizontal=True, key=f"src_{idx}")

    # ── Déterminer image_sel selon la source ──
    image_sel = None
    if source_sel == "📋 Capture active" and has_capture:
        try:
            image_sel = base64_to_pil(st.session_state[cap_key])
            st.markdown('<span class="source-badge">📋 Capture</span>', unsafe_allow_html=True)
        except:
            image_sel = None

    if image_sel is None:
        photo_sel = photos[idx]; photo_sel.seek(0)
        image_sel = Image.open(photo_sel)
        image_sel = corriger_rotation_auto(image_sel, idx)

        rc1, rc2, rc3 = st.columns([1, 1, 2])
        with rc1:
            if st.button("↺ Gauche", key=f"rot_g_{idx}", use_container_width=True):
                current = st.session_state.get(f"rotation_{idx}", 0)
                st.session_state[f"rotation_{idx}"] = (current + 90) % 360
                st.rerun()
        with rc2:
            if st.button("↻ Droite", key=f"rot_d_{idx}", use_container_width=True):
                current = st.session_state.get(f"rotation_{idx}", 0)
                st.session_state[f"rotation_{idx}"] = (current - 90) % 360
                st.rerun()
        with rc3:
            angle = st.session_state.get(f"rotation_{idx}", 0)
            st.caption(f"Rotation : {angle}° | Auto : {st.session_state.get(f'angle_auto_{idx}', 0)}°")

        angle_rot = st.session_state.get(f"rotation_{idx}", 0)
        if angle_rot != 0:
            image_sel = image_sel.rotate(angle_rot, expand=True)

    W_FIXE = 650
    H_FIXE = 500
    img_aff = image_sel.resize((W_FIXE, H_FIXE), Image.LANCZOS)
    W_aff = W_FIXE
    H_aff = H_FIXE
    scale_h = image_sel.height / H_FIXE

    with st.spinner("Détection IA..."):
        img_ann, objets = detecter_objets(image_sel)
    image_zoomable(img_ann, height=500, caption=f"Objets détectés : {len(objets)}")

    st.markdown("**Étape 1 — Trace le bâtiment (rectangle rouge)**")
    st.caption("Dessine du haut du bâtiment jusqu'au sol")

    canvas_bat = st_canvas(
        fill_color="rgba(255,0,0,0.08)",
        stroke_width=3,
        stroke_color="#FF4444",
        background_image=img_aff,
        update_streamlit=True,
        width=W_aff,
        height=H_aff,
        drawing_mode="rect",
        key=f"canvas_bat_{idx}_{st.session_state.get(f'canvas_ver_{idx}', 0)}",
    )

    pixels_batiment = None
    if canvas_bat.json_data and canvas_bat.json_data["objects"]:
        rects = [o for o in canvas_bat.json_data["objects"] if o.get("type") == "rect"]
        if rects:
            dernier = rects[-1]
            h_dessin = abs(dernier.get("height", 0))
            pixels_batiment = int(h_dessin * scale_h)
            st.success(f"Bâtiment tracé : **{pixels_batiment} pixels réels**")

    if pixels_batiment:
        st.markdown("**Étape 2 — Référence**")
        options = []
        for obj in objets:
            options.append({
                "label": f"{obj['label']} (auto) — {obj['pixels']}px = {obj['hauteur_reelle']}m",
                "pixels": obj["pixels"], "hauteur_reelle": obj["hauteur_reelle"], "type": "auto"
            })
        options += [
            {"label": "Porte (manuelle) = 2.2m",   "pixels": None, "hauteur_reelle": 2.2, "type": "manuel"},
            {"label": "Garage (manuel) = 3.0m",    "pixels": None, "hauteur_reelle": 3.0, "type": "manuel"},
            {"label": "Voiture (manuelle) = 1.5m", "pixels": None, "hauteur_reelle": 1.5, "type": "manuel"},
        ]
        labels = [o["label"] for o in options]
        choix = st.selectbox("Référence", labels, key=f"ref_{idx}")
        ref = options[labels.index(choix)]

        if ref["type"] == "manuel":
            couleurs = {
                "Porte (manuelle) = 2.2m":   "#aa44ff",
                "Garage (manuel) = 3.0m":    "#ff8844",
                "Voiture (manuelle) = 1.5m": "#44aaff"
            }
            st.markdown("**Trace la référence**")
            canvas_ref = st_canvas(
                fill_color="rgba(0,0,0,0.05)", stroke_width=3,
                stroke_color=couleurs.get(ref["label"], "#e8c840"),
                background_image=img_aff, update_streamlit=True,
                width=W_aff, height=H_aff, drawing_mode="rect",
                key=f"canvas_ref_{idx}_{choix}_{st.session_state.get(f'canvas_ver_{idx}', 0)}"
            )
            if canvas_ref.json_data and canvas_ref.json_data["objects"]:
                obj_r = canvas_ref.json_data["objects"][-1]
                h_ref_dessin = abs(obj_r.get("height", 0))
                ref["pixels"] = int(h_ref_dessin * scale_h)
                st.success(f"Référence : **{ref['pixels']} px réels** = {ref['hauteur_reelle']} m")

        if ref["pixels"] and pixels_batiment:
            h = (pixels_batiment / ref["pixels"]) * ref["hauteur_reelle"]
            st.session_state[f"hauteur_{idx}"] = h
            r1, r2 = st.columns(2)
            with r1:
                st.markdown(f'<div class="hauteur-result">{h:.2f} m</div>', unsafe_allow_html=True)
            with r2:
                az_disp2 = d["az_arr"] if d["az_arr"] else "—"
                st.markdown(f'<div class="azimut-badge">{az_disp2}°</div>', unsafe_allow_html=True)
            with st.expander("Détails du calcul"):
                st.write(f"**Formule :** ({pixels_batiment} ÷ {ref['pixels']}) × {ref['hauteur_reelle']} = **{h:.2f} m**")
                st.write(f"**Photo :** {d['nom']} — {d['date']}")
                if d['lat']:
                    st.write(f"**GPS :** {d['lat']:.6f}, {d['lon']:.6f}")