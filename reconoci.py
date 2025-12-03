import cv2
import numpy as np
import os
import face_recognition


class Reconocimiento:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.COLOR_PRIMARIO = (70, 70, 70)      # Gris oscuro
        self.COLOR_SECUNDARIO = (100, 100, 100)  # Gris medio
        self.COLOR_EXITO = (80, 200, 120)        # Verde apagado
        self.COLOR_ERROR = (80, 80, 200)         # Rojo apagado
        self.COLOR_ENFOQUE = (200, 200, 200)     # Gris claro
        self.COLOR_FONDO = (255, 255, 255)       # Blanco
        self.COLOR_TEXTO = (255, 255, 255)       # Blanco
        self.COLOR_BORDE = (255, 255, 255)       # Blanco
    def Cargaencodings(self):
        encodings = []
        nombres = []
        for file in os.listdir("rostrosguardados"):
            nombre = os.path.splitext(file)[0]
            encodings = np.load("rostrosguardados/" + file)
            nombres.append(nombre)
        return encodings, nombres