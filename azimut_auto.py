import time
import pyperclip
import pyautogui
import exifread
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ── Configuration ────────────────────────────────────────
DOSSIER_SURVEILLE = r"C:\Users\pc\Downloads"  # Dossier téléchargements
EXTENSIONS        = [".jpg", ".jpeg", ".png"]

def arrondir_azimut(azimut):
    """Arrondit l'azimut à l'ordre de 10"""
    return round(azimut / 10) * 10

def lire_azimut(chemin_photo):
    """Lit l'azimut EXIF de la photo"""
    try:
        with open(chemin_photo, "rb") as f:
            tags = exifread.process_file(f, details=False)
            az = tags.get("GPS GPSImgDirection")
            if az:
                valeur = az.values[0]
                return float(valeur.num) / float(valeur.den)
    except:
        pass
    return None

def traiter_photo(chemin_photo):
    """Traite une nouvelle photo détectée"""
    nom = os.path.basename(chemin_photo)
    print(f"\n📷 Nouvelle photo détectée : {nom}")

    azimut_brut = lire_azimut(chemin_photo)

    if azimut_brut is not None:
        azimut_arrondi = arrondir_azimut(azimut_brut)

        print(f"📐 Azimut brut    : {azimut_brut:.1f}°")
        print(f"🔢 Azimut arrondi : {azimut_arrondi}°")

        # Copier dans le presse-papier
        pyperclip.copy(str(azimut_arrondi))
        print(f"✅ Azimut {azimut_arrondi}° copié dans le presse-papier !")
        print(f"👉 Colle-le dans Zendantennes avec Ctrl+V")

        # Notification sonore
        pyautogui.alert(
            text=f"Photo : {nom}\n\nAzimut brut : {azimut_brut:.1f}°\nAzimut arrondi : {azimut_arrondi}°\n\nCopié dans le presse-papier !\nColle avec Ctrl+V dans Zendantennes",
            title="LNE — Azimut détecté",
            button="OK"
        )
    else:
        print("⚠️ Pas d'azimut dans cette photo")

class SurveilleurPhotos(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        chemin = event.src_path
        ext    = os.path.splitext(chemin)[1].lower()
        if ext in EXTENSIONS:
            time.sleep(1)  # Attendre que le fichier soit complet
            traiter_photo(chemin)

# ── Lancer la surveillance ───────────────────────────────
print("=" * 50)
print("🏗️  LNE — Surveillance Azimut Auto")
print("=" * 50)
print(f"📁 Dossier surveillé : {DOSSIER_SURVEILLE}")
print("⏳ En attente de nouvelles photos...")
print("   (Appuie sur Ctrl+C pour arrêter)")
print("=" * 50)

observer = Observer()
observer.schedule(SurveilleurPhotos(), DOSSIER_SURVEILLE, recursive=False)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
    print("\n⛔ Surveillance arrêtée")

observer.join()