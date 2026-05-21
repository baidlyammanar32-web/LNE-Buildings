from roboflow import Roboflow

rf = Roboflow(api_key="ToqCrk8T5MNyi32eY7mv")

# Chercher dans l'univers public Roboflow
projet = rf.workspace("roboflow-universe-projects").project("door-detection-zpxtg")
dataset = projet.version(1).download("yolov8")

print("Dataset telecharge !")
print(f"Dossier : {dataset.location}")