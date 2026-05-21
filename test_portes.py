from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np
from PIL import Image, ImageOps
import base64
import os

client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="ToqCrk8T5MNyi32eY7mv"
)

def detecter_portes(chemin_photo):
    print(f"Analyse : {chemin_photo}")

    with open(chemin_photo, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    try:
        result = client.infer(
            inference_input=f"data:image/jpeg;base64,{image_b64}",
            model_id="find-door-hvkrh/1"
        )
        print("Resultat :", result)
    except Exception as e:
        print(f"Erreur : {e}")
        return []

    image = Image.open(chemin_photo)
    image = ImageOps.exif_transpose(image)
    image_np  = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    portes = []
    predictions = result.get("predictions", [])
    print(f"{len(predictions)} porte(s) detectee(s)")

    for pred in predictions:
        x    = int(pred["x"])
        y    = int(pred["y"])
        w    = int(pred["width"])
        h    = int(pred["height"])
        conf = pred["confidence"]
        x1, y1 = x - w//2, y - h//2
        x2, y2 = x + w//2, y + h//2

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(image_bgr, f"Porte | {h}px | {conf:.0%}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
        cx = (x1 + x2) // 2
        cv2.arrowedLine(image_bgr, (cx, y2), (cx, y1),
                       (255, 0, 255), 1, tipLength=0.05)
        portes.append(h)
        print(f"Porte : {h}px | confiance : {conf:.0%}")

    image_result = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(image_result).save("resultat_portes.jpg")
    print("Image sauvegardee : resultat_portes.jpg")
    return portes

PHOTO_TEST = r"C:\Users\pc\Desktop\RCP_simulator\nn\VB4137A_2026-04-20_11-29-56.jpg"
if os.path.exists(PHOTO_TEST):
    detecter_portes(PHOTO_TEST)
else:
    print("Change le chemin de la photo !")