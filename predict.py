import tensorflow as tf
import numpy as np
from PIL import Image
import json
import base64
import io
import matplotlib.pyplot  as plt
from tensorflow.keras.preprocessing import image


# Charger le modèle
MODEL_PATH = "model/best_densenet_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)
print("Modèle chargé avec succès")

# Classes des panneaux
CATEGORIES = ["Bird-Drop", "Clean", "Dusty", "Electrical-Damage", "Physical-Damage", "Snow-Covered"]

# Fonction de prédiction
def predict_panel(panel_id, image_source):
    """
    panel_id : ID du panneau (string ou int)
    image_source : chemin d'image ou image encodée en base64
    """

    if isinstance(image_source, str) and image_source.startswith("data:image"):
        header, encoded = image_source.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

   
    else:
        img = Image.open(image_source).convert("RGB")

    # Prétraitement
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prédiction
    preds = model.predict(img_array)
    predicted_index = np.argmax(preds[0])
    predicted_class = CATEGORIES[predicted_index]
    confidence = float(np.max(preds[0]))

    plt.imshow(image.load_img(image_source))
    plt.title(f"Prediction: {predicted_class} ({confidence*100:.2f}%)")
    plt.axis("off")
    plt.show()
    # Réponse JSON standardisée

    return {
        "panel_id": panel_id,
        "predicted_class": predicted_class,
        "confidence": round(confidence * 100, 2),
        "status": "dirty" if predicted_class != "Clean" else "clean"
    }

# Exemple d'utilisation en local
if __name__ == "__main__":
    # Exemple sur une image locale
    result = predict_panel(panel_id="P-12", image_source="images/PD.jpg")

    print("\n--- RESULTAT ---")
    print(json.dumps(result, indent=4, ensure_ascii=False))
