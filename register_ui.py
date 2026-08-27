"""
register_ui.py — Interfaz para registrar 76 fotos y enviarlas al servicio facial.
"""
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import requests
import threading
import time

# Configuración
FACIAL_SERVICE = "http://127.0.0.1:8001"
PERSONA_ID = 5        # ← ID de persona en Servicio_reconocimiento
CI = "9899401"        # ← CI de la persona

POSICIONES = ["centro", "izquierda", "derecha", "arriba", "abajo", "sonrisa"]
CAPTURAS_POR_POSICION = {
    "centro": 10,
    "izquierda": 14,
    "derecha": 14,
    "arriba": 14,
    "abajo": 14,
    "sonrisa": 10,
}
TOTAL_CAPTURAS = sum(CAPTURAS_POR_POSICION.values())


class RegistroFacialApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title(f"Registro Facial - {TOTAL_CAPTURAS} Capturas")

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.ejecutando = False
        self.total_capturas = 0

        # UI
        self.label_video = tk.Label(self.window)
        self.label_video.pack()

        self.label_posicion = tk.Label(self.window, text="Posición: centro", font=("Arial", 16))
        self.label_posicion.pack()

        self.progress = ttk.Progressbar(self.window, length=400, maximum=TOTAL_CAPTURAS)
        self.progress.pack(pady=10)

        self.label_progreso = tk.Label(self.window, text=f"0/{TOTAL_CAPTURAS} capturas")
        self.label_progreso.pack()

        self.label_estado = tk.Label(self.window, text="Presione Iniciar", fg="blue")
        self.label_estado.pack()

        self.btn_iniciar = tk.Button(self.window, text="Iniciar Registro", command=self.iniciar, bg="green", fg="white")
        self.btn_iniciar.pack(pady=5)

        self.actualizar_video()
        self.window.protocol("WM_DELETE_WINDOW", self.cerrar)
        self.window.mainloop()

    def actualizar_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label_video.imgtk = imgtk
            self.label_video.configure(image=imgtk)
        self.window.after(30, self.actualizar_video)

    def iniciar(self):
        self.ejecutando = True
        self.total_capturas = 0
        self.btn_iniciar.config(state="disabled")
        threading.Thread(target=self.proceso_registro, daemon=True).start()

    def proceso_registro(self):
        for i, posicion in enumerate(POSICIONES):
            self.window.after(0, lambda p=posicion, idx=i: self.label_posicion.config(text=f"Posición: {p} ({idx+1}/6)"))
            num_capturas = CAPTURAS_POR_POSICION[posicion]

            for j in range(num_capturas):
                if not self.ejecutando:
                    return

                self.window.after(0, lambda p=posicion, j=j, n=num_capturas: self.label_estado.config(text=f"Capturando {p} {j+1}/{n}..."))
                time.sleep(0.3)

                ret, frame = self.cap.read()
                if not ret:
                    print("No se pudo capturar frame")
                    continue

                _, buffer = cv2.imencode('.jpg', frame)

                try:
                    response = requests.post(
                        f"{FACIAL_SERVICE}/register",
                        files={"image": ("foto.jpg", buffer.tobytes(), "image/jpeg")},
                        data={"persona_id": PERSONA_ID, "ci": CI, "posicion": posicion}
                    )
                    result = response.json()
                    print(f"[{posicion} {j+1}] Respuesta: {result}")

                    if result.get("success"):
                        self.total_capturas += 1
                        self.window.after(0, lambda t=self.total_capturas: self.progress.config(value=t))
                        self.window.after(0, lambda t=self.total_capturas: self.label_progreso.config(text=f"{t}/{TOTAL_CAPTURAS} capturas - Faltan {result.get('faltan', '?')}"))
                    else:
                        print(f"  -> Falló: {result.get('message')}")
                except Exception as e:
                    print(f"Error: {e}")

        self.window.after(0, lambda: self.label_estado.config(text="¡Registro completado!", fg="green"))
        self.window.after(0, lambda: self.btn_iniciar.config(state="normal"))
        self.ejecutando = False

    def cerrar(self):
        self.ejecutando = False
        self.cap.release()
        self.window.destroy()


if __name__ == "__main__":
    RegistroFacialApp()