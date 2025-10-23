import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os

# 1 Chargement du modèle
model_path = "model/best_densenet_model.keras"  
model = tf.keras.models.load_model(model_path)
print("✅ Modèle chargé avec succès")

# 2 Définition des classes 
categories = ["Bird-Drop", "Clean", "Dusty", "Electrical-Damage", "Physical-Damage", "Snow-Covered"]

# 3 Chargement et préparation d'une image à tester

img_path = "images/clean.png"  

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 4 La prédiction

predictions = model.predict(img_array)
predicted_index = np.argmax(predictions[0])
predicted_class = categories[predicted_index]
confidence = np.max(predictions[0]) * 100

#5 L'affichage du résultat

plt.imshow(image.load_img(img_path))
plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%)")
plt.axis("off")
plt.show()

print(f"Classe prédite : {predicted_class} ({confidence:.2f}%)")
