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
from torchvision import transforms, models
import torch
import torch.nn as nn
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
    cap_html = f"<div style='font-size:11px;color:#aaa;text-align:center;margin-top:4px;'>{caption}</div>" if caption else ""
    html_code = f"""
    <div style="position:relative;width:100%;height:{height}px;
                background:#1a252f;border-radius:8px;overflow:hidden;
                border:1px solid #34495e;">
      <canvas id="zc_{hash(data_url)%99999}" style="width:100%;height:100%;cursor:grab;display:block;"></canvas>
      <div style="position:absolute;top:6px;right:8px;font-size:10px;
                  color:#FFD700;background:rgba(26,37,47,0.8);
                  padding:2px 8px;border-radius:8px;pointer-events:none;">
        🖱️ Molette=zoom | Drag=déplacer
      </div>
      <button onclick="rv_{hash(data_url)%99999}()" style="position:absolute;bottom:8px;right:8px;
        background:#FFD700;color:#1a252f;border:none;border-radius:6px;
        font-size:11px;font-weight:bold;padding:4px 10px;cursor:pointer;">↺ Reset</button>
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
            html='<div style="width:36px;height:36px;background:#2c3e50;border-radius:50%;border:3px solid #FFD700;display:flex;align-items:center;justify-content:center;font-size:16px;">📷</div>',
            icon_size=(36,36),icon_anchor=(18,18)),
        popup=folium.Popup(f"<b>Photo</b><br>Azimut:{azimut:.0f}°<br>Lat:{lat:.6f}<br>Lon:{lon:.6f}",max_width=200)
    ).add_to(carte)
    folium.Polygon(locations=[[lat,lon],pt_g,pt_c,pt_d,[lat,lon]],
        color="#FF8C00",fill=True,fill_color="#FFD700",fill_opacity=0.25,weight=2,
        tooltip=f"Direction — {azimut:.0f}°").add_to(carte)
    folium.PolyLine(locations=[[lat,lon],pt_c],color="#FF4500",weight=2,dash_array="6 3").add_to(carte)
    osm_srcdoc=carte._repr_html_().replace("&","&amp;").replace('"',"&quot;")
    gsv_src=f"https://maps.google.com/maps?q={lat},{lon}&layer=c&cbll={lat},{lon}&cbp=12,{azimut:.0f},0,0,0&output=svembed"
    gsv_lien=f"https://www.google.com/maps/@{lat},{lon},3a,80y,{azimut:.0f}h,0t/data=!3m6!1e1"

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#1a252f;font-family:Arial,sans-serif}}
  .panel{{width:100%;position:relative;background:#0d1b2a;border-radius:8px;overflow:hidden;margin-bottom:6px;}}
  .panel-osm{{height:380px;}}
  .panel-sv{{height:320px;}}
  .badge{{position:absolute;top:8px;left:8px;z-index:9999;background:rgba(26,37,47,0.92);color:#FFD700;font-size:11px;font-weight:bold;padding:3px 10px;border-radius:10px;pointer-events:none;}}
  iframe{{width:100%;height:100%;border:none}}
  .open-btn{{position:absolute;bottom:10px;right:10px;z-index:9999;background:#FFD700;color:#1a252f;font-size:11px;font-weight:bold;padding:6px 12px;border-radius:6px;text-decoration:none;}}
</style></head><body>
<div class="panel panel-osm">
  <div class="badge">🗺️ Carte OSM</div>
  <iframe srcdoc="{osm_srcdoc}"></iframe>
</div>
<div class="panel panel-sv">
  <div class="badge">🚶 Street View — {azimut:.0f}°</div>
  <iframe src="{gsv_src}" allowfullscreen loading="lazy"></iframe>
  <a class="open-btn" href="{gsv_lien}" target="_blank">↗ Ouvrir</a>
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
                cv2.rectangle(img_bgr,(x1,y1),(x2,y2),(0,200,0),2)
                cv2.putText(img_bgr,f"Voiture|{px}px",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,200,0),2)
                cv2.arrowedLine(img_bgr,((x1+x2)//2,y2),((x1+x2)//2,y1),(0,200,0),1,tipLength=0.05)
                objets.append({"label":"Voiture","pixels":px,"hauteur_reelle":1.5})
    modele_p=charger_modele_portes()
    if modele_p:
        try:
            res_p=modele_p(img_bgr,verbose=False,conf=0.3)
            for r in res_p:
                for box in r.boxes:
                    x1,y1,x2,y2=map(int,box.xyxy[0]); conf=float(box.conf[0]); px=y2-y1
                    cv2.rectangle(img_bgr,(x1,y1),(x2,y2),(255,0,255),2)
                    cv2.putText(img_bgr,f"Porte|{px}px|{conf:.0%}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,0,255),2)
                    cv2.arrowedLine(img_bgr,((x1+x2)//2,y2),((x1+x2)//2,y1),(255,0,255),1,tipLength=0.05)
                    objets.append({"label":"Porte","pixels":px,"hauteur_reelle":2.2})
        except: pass
    return Image.fromarray(cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)),objets

# ════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════
st.set_page_config(page_title="LNE — Analyse Bâtiments",layout="wide",initial_sidebar_state="collapsed")
st.markdown("""<style>
  .stApp,section[data-testid="stMain"]{background:#1a252f}
  .lne-header{background:#FFD700;padding:8px 16px;border-radius:8px;margin-bottom:10px}
  .lne-header span{font-size:20px;font-weight:bold;color:#1a252f}
  .sec-title{font-size:12px;font-weight:700;color:#FFD700;border-bottom:2px solid #FFD700;padding-bottom:3px;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px}
  .meta{font-size:10px;color:#aaa;margin-top:4px;font-family:monospace}
  .meta b{color:#FFD700}
  .hauteur-badge{display:inline-block;background:#27ae60;color:white;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:bold;margin-top:4px}
  .hauteur-result{background:#27ae60;color:white;border-radius:10px;padding:12px;text-align:center;font-size:28px;font-weight:bold;margin-top:8px}
  .azimut-badge{background:#FFD700;color:#1a252f;border-radius:8px;padding:8px;text-align:center;font-size:32px;font-weight:bold}
  .source-badge{display:inline-block;background:#8e44ad;color:white;padding:3px 10px;border-radius:10px;font-size:11px;font-weight:bold;margin-bottom:6px}
  #MainMenu,footer,header{visibility:hidden}
  .block-container{padding-top:0.5rem!important}
  ::-webkit-scrollbar{width:6px;height:6px}
  ::-webkit-scrollbar-track{background:#1a252f}
  ::-webkit-scrollbar-thumb{background:#FFD700;border-radius:3px}
  /* Style du bouton paste */
  div[data-testid="stButton"] > button[kind="secondary"] {
    background:#2c3e50 !important;
    border:2px dashed #FFD700 !important;
    color:#FFD700 !important;
    font-weight:bold !important;
    width:100% !important;
    padding:16px !important;
    border-radius:10px !important;
  }
</style>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════
st.markdown("""<div class="lne-header">
  <span>🏗️ LNE — Analyse de Bâtiments &nbsp;
    <span style="background:#1a252f;color:white;padding:3px 10px;border-radius:5px;font-size:13px;">Zendantennes</span>
  </span></div>""", unsafe_allow_html=True)

col_f1,_=st.columns([2,6])
with col_f1:
    st.text_input("",value="File 00000001",placeholder="Numéro de dossier",label_visibility="collapsed")

# ════════════════════════════════════════════════════════
#  UPLOAD
# ════════════════════════════════════════════════════════
photos=st.file_uploader("Photos",type=["jpg","jpeg","png"],accept_multiple_files=True,label_visibility="collapsed")
if not photos:
    st.markdown('<div style="text-align:center;padding:60px;color:#aaa;"><div style="font-size:48px;">📁</div><div style="font-size:16px;margin-top:12px;">Importe tes photos pour commencer</div></div>',unsafe_allow_html=True)
    st.stop()

# ════════════════════════════════════════════════════════
#  EXIF
# ════════════════════════════════════════════════════════
photos_data=[]
for photo in photos:
    photo.seek(0); lat,lon=extraire_gps(photo)
    photo.seek(0); azimut=extraire_azimut(photo)
    photo.seek(0); date=extraire_date(photo)
    az_arr=round(azimut/10)*10 if azimut else None
    photos_data.append({"nom":photo.name,"lat":lat,"lon":lon,"azimut":azimut,"az_arr":az_arr,"date":date})

if "idx_sel" not in st.session_state:
    st.session_state.idx_sel=0

col_g,col_d=st.columns([5,3],gap="small")

# ════════════════════════════════════════════════════════
#  DROITE — grille photos
# ════════════════════════════════════════════════════════
with col_d:
    st.markdown('<div class="sec-title">📷 Photos</div>',unsafe_allow_html=True)
    n1,n2,n3=st.columns([1,2,1])
    with n1:
        if st.button("◀",use_container_width=True):
            if st.session_state.idx_sel>0: st.session_state.idx_sel-=1; st.rerun()
    with n2:
        st.markdown(f"<div style='text-align:center;color:#aaa;font-size:11px;padding-top:6px;'>{st.session_state.idx_sel+1} / {len(photos)}</div>",unsafe_allow_html=True)
    with n3:
        if st.button("▶",use_container_width=True):
            if st.session_state.idx_sel<len(photos)-1: st.session_state.idx_sel+=1; st.rerun()

    for row in range(0,len(photos),2):
        cols=st.columns(2,gap="small")
        for ci in range(2):
            idx=row+ci
            if idx>=len(photos): break
            d=photos_data[idx]; photo=photos[idx]
            photo.seek(0)
            img=Image.open(photo)
            img=corriger_rotation_auto(img,idx)
            img.thumbnail((160,110))
            with cols[ci]:
                est_sel=st.session_state.idx_sel==idx
                border="2px solid #e74c3c" if est_sel else "2px solid #34495e"
                st.markdown(f'<div style="background:#2c3e50;border-radius:8px;padding:5px;border:{border};margin-bottom:6px;">',unsafe_allow_html=True)
                st.image(img,use_container_width=True)
                az_txt=f"{d['az_arr']}°" if d['az_arr'] else "—"
                lon_txt=f"{d['lon']:.2f}" if d['lon'] else "N/A"
                lat_txt=f"{d['lat']:.2f}" if d['lat'] else "N/A"
                st.markdown(f'<div class="meta"><b>Date</b> {d["date"]}<br><b>Az</b> {az_txt} &nbsp;<b>X</b> {lon_txt} &nbsp;<b>Y</b> {lat_txt}</div>',unsafe_allow_html=True)
                if f"hauteur_{idx}" in st.session_state:
                    st.markdown(f'<span class="hauteur-badge">⬆ {st.session_state[f"hauteur_{idx}"]:.2f} m</span>',unsafe_allow_html=True)
                st.markdown('</div>',unsafe_allow_html=True)
                if st.button("Voir",key=f"btn_{idx}",use_container_width=True):
                    st.session_state.idx_sel=idx; st.rerun()

# ════════════════════════════════════════════════════════
#  GAUCHE — Carte + Capture + Calcul
# ════════════════════════════════════════════════════════
with col_g:
    idx=st.session_state.idx_sel
    d=photos_data[idx]
    cap_key=f"capture_data_{idx}"

    # Carte
    st.markdown(f'<div class="sec-title">🗺️ {d["nom"]} — Az {d["az_arr"]}°</div>',unsafe_allow_html=True)
    carte_html=afficher_carte(d)
    if carte_html:
        html(carte_html,height=720)
    else:
        st.warning("Pas de données GPS pour cette photo")

    # ════════════════════════════════════════════
    #  CAPTURE — Coller depuis presse-papiers
    # ════════════════════════════════════════════
    st.markdown('<div class="sec-title">📸 Capture d\'écran</div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:11px;color:#aaa;margin-bottom:8px;'>"
        "1️⃣ Fais ta capture (PrintScreen) &nbsp; "
        "2️⃣ Clique le bouton ci-dessous &nbsp; "
        "3️⃣ L'image apparaît automatiquement"
        "</div>",
        unsafe_allow_html=True
    )

    # Bouton coller — streamlit-paste-button gère le clipboard natif
    paste_result = paste_image_button(
        label="📋  Coller depuis le presse-papiers (Ctrl+V)",
        key=f"paste_{idx}",
        background_color="#2c3e50",
        hover_background_color="#34495e",
        text_color="#FFD700",
    )

    # Si une image vient d'être collée, on la stocke
    if paste_result.image_data is not None:
        img_pasted = paste_result.image_data.convert("RGB")
        st.session_state[cap_key] = pil_to_base64(img_pasted)
        st.success("✅ Capture collée avec succès !")

    # Upload manuel fallback
    st.markdown("<div style='font-size:11px;color:#aaa;margin-top:6px;margin-bottom:2px;'>Ou importe un fichier :</div>", unsafe_allow_html=True)
    upload_cap = st.file_uploader(
        "Importer une capture",
        type=["jpg","jpeg","png"],
        key=f"upload_{idx}",
        label_visibility="collapsed"
    )
    if upload_cap:
        img_up = Image.open(upload_cap).convert("RGB")
        st.session_state[cap_key] = pil_to_base64(img_up)
        st.success("✅ Capture importée !")

    # Bouton effacer
    if st.session_state.get(cap_key):
        if st.button("🗑️ Effacer la capture active", key=f"eff_{idx}"):
            st.session_state.pop(cap_key, None)
            st.rerun()

    # Aperçu capture active
    has_capture = bool(st.session_state.get(cap_key, ""))
    if has_capture:
        img_prev = base64_to_pil(st.session_state[cap_key])
        image_zoomable(img_prev, height=250, caption="📋 Capture active")

    # ════════════════════════════════════════════
    #  CALCUL HAUTEUR
    # ════════════════════════════════════════════
    st.markdown('<div class="sec-title">📐 Calcul Hauteur</div>', unsafe_allow_html=True)

    # Source image
    source_opts = ["🖼️ Photo originale"]
    if has_capture:
        source_opts.insert(0, "📋 Capture active")
    source_sel = st.radio("Source :", source_opts, horizontal=True, key=f"src_{idx}")

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
        rc1, rc2 = st.columns([2, 3])
        with rc1:
            rot = st.selectbox("Rotation", [0, 90, 180, 270], key=f"rot_{idx}")
            if rot != 0:
                st.session_state[f"rotation_{idx}"] = rot
                image_sel = image_sel.rotate(rot, expand=True)
        with rc2:
            st.caption(f"Auto : {st.session_state.get(f'angle_auto_{idx}', 0)}°")

    # Resize
    W0, H0 = image_sel.size
    scale = min(700/W0, 900/H0, 1.0)
    W_aff = int(W0*scale); H_aff = int(H0*scale)
    img_aff = image_sel.resize((W_aff, H_aff), Image.LANCZOS)

    # YOLO
    with st.spinner("Détection IA..."):
        img_ann, objets = detecter_objets(image_sel)
    image_zoomable(img_ann, height=500, caption=f"Objets : {len(objets)}")

    # Étape 1 — Canvas bâtiment
    st.markdown("**Étape 1 — Trace le bâtiment**")
    canvas_bat = st_canvas(
        fill_color="rgba(255,0,0,0.1)", stroke_width=3, stroke_color="#FF0000",
        background_image=img_aff, update_streamlit=True,
        width=W_aff, height=H_aff, drawing_mode="rect",
        key=f"canvas_bat_{idx}"
    )

    pixels_batiment = None
    if canvas_bat.json_data and canvas_bat.json_data["objects"]:
        dernier = canvas_bat.json_data["objects"][-1]
        pixels_batiment = abs(int(dernier["height"]/scale))
        st.success(f"Bâtiment : **{pixels_batiment} pixels**")

    # Étape 2 — Référence
    if pixels_batiment:
        st.markdown("**Étape 2 — Référence**")
        options = []
        for obj in objets:
            options.append({
                "label": f"{obj['label']} (auto) — {obj['pixels']}px = {obj['hauteur_reelle']}m",
                "pixels": obj["pixels"],
                "hauteur_reelle": obj["hauteur_reelle"],
                "type": "auto"
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
                "Porte (manuelle) = 2.2m":   "#AA00FF",
                "Garage (manuel) = 3.0m":    "#FF8800",
                "Voiture (manuelle) = 1.5m": "#00AAFF"
            }
            st.markdown("**Trace la référence**")
            canvas_ref = st_canvas(
                fill_color="rgba(0,0,0,0.05)", stroke_width=3,
                stroke_color=couleurs.get(ref["label"], "#FF0000"),
                background_image=img_aff, update_streamlit=True,
                width=W_aff, height=H_aff, drawing_mode="rect",
                key=f"canvas_ref_{idx}_{choix}"
            )
            if canvas_ref.json_data and canvas_ref.json_data["objects"]:
                obj_r = canvas_ref.json_data["objects"][-1]
                ref["pixels"] = abs(int(obj_r["height"]/scale))
                st.success(f"Référence : **{ref['pixels']} px** = {ref['hauteur_reelle']} m")

        # Résultat
        if ref["pixels"] and pixels_batiment:
            h = (pixels_batiment / ref["pixels"]) * ref["hauteur_reelle"]
            st.session_state[f"hauteur_{idx}"] = h
            r1, r2 = st.columns(2)
            with r1:
                st.markdown(f'<div class="hauteur-result">{h:.2f} m</div>', unsafe_allow_html=True)
            with r2:
                az_disp = d["az_arr"] if d["az_arr"] else "—"
                st.markdown(f'<div class="azimut-badge">{az_disp}°</div>', unsafe_allow_html=True)
            with st.expander("Détails"):
                st.write(f"**Formule :** ({pixels_batiment} ÷ {ref['pixels']}) × {ref['hauteur_reelle']} = **{h:.2f} m**")
                st.write(f"**Photo :** {d['nom']} — {d['date']}")
                if d['lat']:
                    st.write(f"**GPS :** {d['lat']:.6f}, {d['lon']:.6f}")