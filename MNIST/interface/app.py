import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import tkinter as tk
from PIL import Image, ImageOps, ImageDraw
import numpy as np
from model.predict import predict_digit

window = tk.Tk()
window.title("Reconnaissance MNIST")

canvas = tk.Canvas(window, width=280, height=280, bg='white')
canvas.pack()

image = Image.new("L", (280, 280), "white")
draw = ImageDraw.Draw(image)


def draw_digit(event):
    x, y = event.x, event.y
    r = 10
    canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')
    draw.ellipse([x-r, y-r, x+r, y+r], fill="black")


canvas.bind("<B1-Motion>", draw_digit)


def preprocess_for_mnist(img):
    # inversion (noir devient blanc et inversement)
    img = ImageOps.invert(img)

    # conversion tableau numpy
    arr = np.array(img)

    # seuil pour détection du chiffre
    thresh = arr > 20
    coords = np.argwhere(thresh)

    if coords.size == 0:
        return np.zeros((28, 28))  # aucune écriture détectée

    # délimitation du chiffre
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # crop au plus juste
    arr = arr[y_min:y_max+1, x_min:x_max+1]

    # redimension en 20×20 (comme MNIST)
    img20 = Image.fromarray(arr).resize((20, 20), Image.LANCZOS)
    arr20 = np.array(img20)

    # nouvelle image 28×28 vide
    final = np.zeros((28, 28))

    # centrage
    y_offset = (28 - 20) // 2
    x_offset = (28 - 20) // 2

    final[y_offset:y_offset+20, x_offset:x_offset+20] = arr20

    # normalisation
    final = final / 255.0

    return final


def predict():
    # copie de l'image du canvas (280x280)
    img = image.copy()

    # réduction en 28x28, niveaux de gris
    img = img.resize((28, 28), Image.LANCZOS).convert("L")

    # inversion : fond noir, chiffre blanc (comme MNIST)
    img = ImageOps.invert(img)

    # conversion en array + normalisation
    arr = np.array(img).astype("float32") / 255.0

    # reshape pour le modèle : (1, 28, 28, 1)
    arr = arr.reshape(1, 28, 28, 1)

    # prédiction
    digit = predict_digit(arr)
    result_label.config(text=f"Prédiction : {digit}")



def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, 280, 280], fill="white")


tk.Button(window, text="Prédire", command=predict).pack()
tk.Button(window, text="Effacer", command=clear_canvas).pack()

result_label = tk.Label(window, text="Prédiction :", font=("Arial", 20))
result_label.pack()

window.mainloop()
