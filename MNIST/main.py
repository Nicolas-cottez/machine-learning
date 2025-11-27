import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models

# -----------------------------------------------------------
# 1. Construction du modèle CNN (intégré dans main)
# -----------------------------------------------------------
def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), activation='relu', padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# -----------------------------------------------------------
# 2. Entraînement complet du modèle (intégré dans main)
# -----------------------------------------------------------
def train_model():
    # chargement MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # normalisation et reshape
    x_train = (x_train / 255.0).reshape(-1, 28, 28, 1)
    x_test = (x_test / 255.0).reshape(-1, 28, 28, 1)

    # construction modèle
    model = build_cnn()

    # entraînement
    model.fit(x_train, y_train, epochs=5, batch_size=64)

    # évaluation
    print("Évaluation :")
    model.evaluate(x_test, y_test)

    # sauvegarde
    model.save("./data/mnist_cnn.h5")
    print("Modèle sauvegardé → data/mnist_cnn.h5")


# -----------------------------------------------------------
# 3. Point d'entrée
# -----------------------------------------------------------
if __name__ == "__main__":
    train_model()
