import cv2
import numpy as np
import os
import face_recognition

class Reconocimiento:
    def __init__(self):
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
        if not os.path.exists("rostrosguardados"):
            os.makedirs("rostrosguardados")
            
        for file in os.listdir("rostrosguardados"):
            if file.endswith(".npy"):
                nombre = os.path.splitext(file)[0]
                loaded_encoding = np.load(os.path.join("rostrosguardados", file))
                encodings.append(loaded_encoding)
                nombres.append(nombre)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al de reconocimiento")
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #Buscar rostros
            ubicaciones = face_recognition.face_locations(rgb_frame, model="hog")
            #Codificar rostros
            codificaciones = face_recognition.face_encodings(rgb_frame, ubicaciones)
            
            #Recorrer rostros   
            for ubicar, codificar in zip(ubicaciones, codificaciones):
                #Codificar rostros
                # Default to unknown
                nombre_coincidencia = "Desconocido"
                
                if encodings:
                    coincidencia = face_recognition.compare_faces(encodings, codificar, tolerance=0.6)
                    if True in coincidencia:
                        nombre_coincidencia = nombres[coincidencia.index(True)].upper()
                
                #dibujar rectangulo
                top, right, bottom, left = ubicar
                cv2.rectangle(frame, (left, top), (right, bottom), self.COLOR_EXITO, 2)
                cv2.putText(frame, nombre_coincidencia, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_EXITO, 2)
            
            #Mostrar frame
            cv2.imshow("Reconocimiento", frame)
            
            #Salir
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Sistema Cerrado")
def main():
    reco = Reconocimiento()
    try:
        reco.Cargaencodings()
    except Exception as e:
        print("Error al cargar los encodings:", e)
if __name__ == "__main__":
    main()
            
                
                
