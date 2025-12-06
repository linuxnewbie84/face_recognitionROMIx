import cv2 
import numpy as np
import os
import face_recognition
import json

os.makedirs("rostrosguardados", exist_ok=True)

class Capturar:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
    def capturar(self):
        while True:
            print("¿Qué desea hacer?")
            print("1. Capturar rostro")
            print("2. Salir")
            opcion = input("Opcion: ")
            if opcion == "1":
                nombre = input("Agrega el nombre de la persona que desea guardar:")
                while True:
                    ret, frame = self.cap.read()
                    cv2.imshow("Sistema de Captura de rostros", frame)
                    cv2.putText(frame, "Presione 'C' para capturar  |  'Q' para salir", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    if not ret:
                        print("Error al tomar la foto")
                        break
                    if cv2.waitKey(1) & 0xFF == ord('c'):
                        # Convertir a face_recognition
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # MEJORA: Detectar rostros primero
                        # model="hog" es más rápido (CPU), model="cnn" es más preciso (GPU)
                        ubicar = face_recognition.face_locations(rgb_frame, model="hog")
                        
                        # Pasar las ubicaciones encontradas para no volver a detectar
                        encodings = face_recognition.face_encodings(rgb_frame, ubicar)

                        if encodings:
                            # Guardar solo el primer encoding encontrado
                            encoding = encodings[0]
                            np.save(f"rostrosguardados/{nombre}.npy", encoding)
                            print("Encoding guardado exitosamente")
                            
                            # Guardar en json (append)
                            if os.path.exists("encodings.json"):
                                with open("encodings.json", "r") as f:
                                    try:
                                        data = json.load(f)
                                    except json.JSONDecodeError:
                                        data = {}
                            else:
                                data = {}
                                
                            data[nombre] = encoding.tolist()
                            
                            with open("encodings.json", "w") as f:
                                json.dump(data, f)
                        else:
                            print("No se pudo obtener el encoding")

                        if ubicar:
                            top, right, bottom, left = ubicar[0]
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            cv2.putText(frame, "Rostro detectado", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.imwrite(f"rostrosguardados/{nombre}.jpg", frame)
                            print("Foto guardada exitosamente")
                        else:
                            print("No se pudo obtener el rostro")
                            break                    
                    elif cv2.waitKey(1) & 0xFF == ord('q'):
                        break   
            
        self.cap.release()
        cv2.destroyAllWindows()
        print("Sistema Cerrado")


        
def main():
    cap = Capturar()
    cap.capturar()
    
if __name__ == "__main__":
    main()