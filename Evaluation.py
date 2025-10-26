import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------------------
# 1) Chargement du modèle
# --------------------------------------
MODEL_PATH = "model/best_densenet_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Modèle chargé avec succès\n")

# --------------------------------------
# 2) Chargement du dataset (validation)
# --------------------------------------
BASE_DIR = "dataset/Faulty_solar_panel"  # ⚠️ Modifier selon votre chemin dataset

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_gen = val_datagen.flow_from_directory(
    BASE_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    subset='validation'
)

class_names = list(val_gen.class_indices.keys())
print("\nClasses :", class_names)

# --------------------------------------
# 3) Prédictions sur l'ensemble de validation
# --------------------------------------
print("\n⏳ Prédiction sur les images de validation...")
pred_probs = model.predict(val_gen)
y_pred = np.argmax(pred_probs, axis=1)
y_true = val_gen.classes

# --------------------------------------
# 4) Matrice de Confusion
# --------------------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Prédiction")
plt.ylabel("Vérité terrain")
plt.title("Matrice de Confusion - DenseNet169")
plt.show()

# --------------------------------------
# 5) Rapport de Classification
# --------------------------------------
print("\n📊 Rapport de classification :\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# --------------------------------------
# 6) Grad-CAM (exemple visuel)
# --------------------------------------
def grad_cam_display(img_path):
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import Model
    import cv2

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    last_conv_layer = model.get_layer("conv5_block32_concat")
    grad_model = Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_output)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(tf.multiply(weights, conv_output), axis=-1).numpy()[0]

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (224, 224))

    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    original = cv2.imread(img_path)
    original = cv2.resize(original, (224, 224))
    result = cv2.addWeighted(original, 0.5, heatmap, 0.5, 0)

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(original[..., ::-1])
    plt.title("Image originale")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(result[..., ::-1])
    plt.title("Grad-CAM (zones importantes)")
    plt.axis("off")
    plt.show()

# Exemple d'utilisation Grad-CAM
example_img = val_gen.filepaths[5]
print("\n🎯 Affichage Grad-CAM sur :", example_img)
grad_cam_display(example_img)
