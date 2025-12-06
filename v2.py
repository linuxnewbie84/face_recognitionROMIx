import cv2 
import numpy as np
import os
import face_recognition
from datetime import datetime

os.makedirs("rostrosguardados", exist_ok=True)

class capturar:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.COLOR_PRIMARIO = (70, 70, 70)      # Gris oscuro
        self.COLOR_SECUNDARIO = (100, 100, 100)  # Gris medio
        self.COLOR_EXITO = (80, 200, 120)        # Verde apagado
        self.COLOR_ERROR = (80, 80, 200)         # Rojo apagado
        self.COLOR_ENFOQUE = (200, 200, 200)     # Gris claro
       
        
    def dibujar_interfaz(self, frame, rostro_detectado=False, mensaje=""):
        """Dibuja la interfaz sobre el frame"""
        h, w = frame.shape[:2]
        # Overlay semi-transparente
        overlay = frame.copy()
        
        # Barra superior
        cv2.rectangle(overlay, (0, 0), (w, 80), self.COLOR_PRIMARIO, -1)
        cv2.putText(overlay, "SISTEMA DE CAPTURA FACIAL", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Recuadro de enfoque central para rostro
        centro_x, centro_y = w // 2, h // 2
        ancho_recuadro, alto_recuadro = 350, 450
        x1 = centro_x - ancho_recuadro // 2
        y1 = centro_y - alto_recuadro // 2
        x2 = centro_x + ancho_recuadro // 2
        y2 = centro_y + alto_recuadro // 2
        
        # Color del recuadro según detección
        color_recuadro = self.COLOR_EXITO if rostro_detectado else self.COLOR_ENFOQUE
        grosor = 3 if rostro_detectado else 2
        
        # Dibujar recuadro con esquinas
        longitud_esquina = 40
        cv2.line(overlay, (x1, y1), (x1 + longitud_esquina, y1), color_recuadro, grosor)
        cv2.line(overlay, (x1, y1), (x1, y1 + longitud_esquina), color_recuadro, grosor)
        
        cv2.line(overlay, (x2, y1), (x2 - longitud_esquina, y1), color_recuadro, grosor)
        cv2.line(overlay, (x2, y1), (x2, y1 + longitud_esquina), color_recuadro, grosor)
        
        cv2.line(overlay, (x1, y2), (x1 + longitud_esquina, y2), color_recuadro, grosor)
        cv2.line(overlay, (x1, y2), (x1, y2 - longitud_esquina), color_recuadro, grosor)
        
        cv2.line(overlay, (x2, y2), (x2 - longitud_esquina, y2), color_recuadro, grosor)
        cv2.line(overlay, (x2, y2), (x2, y2 - longitud_esquina), color_recuadro, grosor)
        
        # Texto guía
        texto_guia = "ROSTRO DETECTADO" if rostro_detectado else "POSICIONE SU ROSTRO"
        cv2.putText(overlay, texto_guia, (centro_x - 150, y1 - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_recuadro, 2)
        
        # Barra inferior con instrucciones
        cv2.rectangle(overlay, (0, h - 100), (w, h), self.COLOR_PRIMARIO, -1)
        cv2.putText(overlay, "Presione 'C' para capturar  |  'Q' para salir", 
                    (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Mensaje de estado
        if mensaje:
            color_msg = self.COLOR_EXITO if "exitosa" in mensaje.lower() else self.COLOR_ERROR
            cv2.putText(overlay, mensaje, (20, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_msg, 2)
        
        # Mezclar overlay
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        return frame, (x1, y1, x2, y2)
    
    def detectar_rostro(self, frame):
        """Detecta rostros en el frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ubicaciones_rostros = face_recognition.face_locations(rgb_frame)
        return ubicaciones_rostros
    
    def recortar_rostro(self, frame, ubicacion_rostro):
        """Recorta el rostro del frame con margen"""
        top, right, bottom, left = ubicacion_rostro
        
        # Agregar margen
        margen_h = int((bottom - top) * 0.3)
        margen_w = int((right - left) * 0.2)
        
        top = max(0, top - margen_h)
        bottom = min(frame.shape[0], bottom + margen_h // 2)
        left = max(0, left - margen_w)
        right = min(frame.shape[1], right + margen_w)
        
        return frame[top:bottom, left:right]
    
    def capturar_rostro(self):
        """Proceso de captura de rostro"""
        print("\n" + "="*60)
        print("SISTEMA DE CAPTURA FACIAL".center(60))
        print("="*60)
        nombre = input("\nIngrese el nombre de la persona: ").strip()
        
        if not nombre:
            print("Error: Debe ingresar un nombre valido")
            return
        
        print(f"\nIniciando captura para: {nombre}")
        print("   - Posicione su rostro dentro del recuadro")
        print("   - Presione 'C' cuando este listo")
        print("   - Presione 'Q' para cancelar\n")
        
        mensaje = ""
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error al acceder a la cámara")
                break
            
            frame = cv2.flip(frame, 1)  # Espejo
            
            # Detectar rostro cada 3 frames (optimización)
            rostro_detectado = False
            ubicaciones_rostros = []
            
            if frame_count % 3 == 0:
                ubicaciones_rostros = self.detectar_rostro(frame)
                rostro_detectado = len(ubicaciones_rostros) > 0
            
            # interfaz
            frame, recuadro = self.dibujar_interfaz(frame, rostro_detectado, mensaje)
            
            # Mostrar ubicación del rostro detectado
            if rostro_detectado:
                for (top, right, bottom, left) in ubicaciones_rostros:
                    cv2.rectangle(frame, (left, top), (right, bottom), 
                                self.COLOR_EXITO, 2)
            
            cv2.imshow("Sistema de Captura", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') or key == ord('C'):
                if rostro_detectado:
                    # Guardar imagen completa
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # Recortar y guardar rostro
                    rostro_recortado = self.recortar_rostro(frame, ubicaciones_rostros[0])
                    nombre_rostro = os.path.join("rostrosguardados", 
                                                f"{nombre}_{timestamp}_rostro.jpg")
                    cv2.imwrite(nombre_rostro, rostro_recortado)
                    
                    # Guardar encoding
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb_frame, 
                                                               ubicaciones_rostros)
                    
                    if encodings:
                        nombre_npy = os.path.join("rostrosguardados", 
                                                 f"{nombre}_{timestamp}.npy")
                        np.save(nombre_npy, encodings[0])
                        mensaje = f" Captura exitosa: {nombre}"
                        print(f"\nRostro capturado y guardado exitosamente")
                        print(f"Rostro: {nombre_rostro}")
                        print(f"Encoding: {nombre_npy}")
                    else:
                        mensaje = " Error al procesar encoding"
                        print(" No se pudo generar el encoding facial")
                    
                    # Mostrar mensaje por 2 segundos
                    for _ in range(60):
                        ret, frame = self.cap.read()
                        frame = cv2.flip(frame, 1)
                        frame, _ = self.dibujar_interfaz(frame, False, mensaje)
                        cv2.imshow("Sistema de Captura", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    break
                else:
                    mensaje = "No se detecta rostro"
            
            elif key == ord('q') or key == ord('Q'):
                print(" Captura cancelada")
                break
            
            frame_count += 1
        
        cv2.destroyWindow("Sistema de Captura")
    
    def menu_principal(self):
        """Menú principal del sistema"""
        while True:
            print("\n" + "="*60)
            print("SISTEMA DE CAPTURA FACIAL".center(60))
            print("="*60)
            print("\n  [1] Capturar nuevo rostro")
            print("  [2] Ver rostros guardados")
            print("  [3] Salir del sistema")
            print("\n" + "-"*60)
            
            opcion = input("\n→ Seleccione una opción: ").strip()
            
            if opcion == "1":
                self.capturar_rostro()
            elif opcion == "2":
                self.listar_rostros()
            elif opcion == "3":
                print("\n" + "="*60)
                print("Sistema finalizado. ¡Hasta pronto!".center(60))
                print("="*60 + "\n")
                break
            else:
                print("\n Opción no válida. Intente nuevamente.")
    
    def listar_rostros(self):
        """Lista los rostros guardados"""
        archivos = sorted([f for f in os.listdir("rostrosguardados") if f.endswith('.jpg')])
        
        print("\n" + "="*60)
        print("ROSTROS GUARDADOS".center(60))
        print("="*60)
        
        if not archivos:
            print("\n No hay rostros guardados todavía")
        else:
            print(f"\n  Total de capturas: {len(archivos)}\n")
            for i, archivo in enumerate(archivos, 1):
                base_name = archivo.replace("_rostro.jpg", "")
                npy_file = f"{base_name}.npy"
                npy_path = os.path.join("rostrosguardados", npy_file)
                
                print(f"  {i}. {archivo}")
                
                if os.path.exists(npy_path):
                    try:
                        datos = np.load(npy_path)
                        print(f"     Encoding shape: {datos.shape}")
                        # print(datos) # Opcional: imprimir datos crudos
                    except Exception as e:
                        print(f"     Error al leer .npy: {e}")
                else:
                    print(f"     (No se encontró archivo .npy: {npy_file})")
        
        print("\n" + "-"*60)
        input("\nPresione ENTER para continuar...")
    
    def cerrar(self):
        """Liberar recursos"""
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    sistema = capturar()
    try:
        sistema.menu_principal()
    except KeyboardInterrupt:
        print("\n\nSistema interrumpido por el usuario")
    finally:
        sistema.cerrar()

if __name__ == "__main__":
    main()