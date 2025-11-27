import numpy as np
from tensorflow.keras.models import load_model

# Chargement du modèle une seule fois
model = load_model("./data/mnist_cnn.h5")

def predict_digit(img_tensor):
    """
    img_tensor : array numpy de forme (1, 28, 28, 1)
    déjà normalisé entre 0 et 1.
    """
    pred = model.predict(img_tensor)
    return int(np.argmax(pred))
