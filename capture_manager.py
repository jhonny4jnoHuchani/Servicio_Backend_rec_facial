"""
capture_manager.py — Gestor de capturas de verificación.
Guarda imágenes organizadas por CI y fecha.
Solo guarda la foto frontal.
"""
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from config import CAPTURES_DIR, SAVE_CAPTURES_ENABLED, SAVE_RECONOCIDO


class CaptureManager:
    """
    Gestiona el guardado de imágenes de verificación.
    Estructura: capturas/{CI}/{fecha_dd-mm-aaaa_Día}/{tipo}_{resultado}_{hora}.jpg
    """
    
    # Mapeo de resultado → nombre para el archivo
    RESULTADOS = {
        "reconocido": "reconocido",
        "desconocido": "desconocido",
        "gesto_no_coincide": "gesto_no_coincide",
        "spoofing_detectado": "spoofing",
        "con_lentes": "con_lentes",
        "error": "error"
    }
    
    # Días de la semana en español
    DIAS_SEMANA = {
        0: "Lunes",
        1: "Martes",
        2: "Miércoles",
        3: "Jueves",
        4: "Viernes",
        5: "Sábado",
        6: "Domingo"
    }
    
    def __init__(self):
        self.enabled = SAVE_CAPTURES_ENABLED
        self.save_reconocido = SAVE_RECONOCIDO
        self.base_dir = CAPTURES_DIR
        print(f"[CAPTURE] Gestor de capturas inicializado (enabled={self.enabled})", flush=True)
    
    def _get_fecha_carpeta(self) -> str:
        """
        Retorna el nombre de la carpeta de fecha.
        Formato: dd-mm-aaaa_Día
        Ejemplo: 01-09-2026_Martes
        """
        now = datetime.now()
        fecha = now.strftime("%d-%m-%Y")
        dia_semana = self.DIAS_SEMANA[now.weekday()]
        return f"{fecha}_{dia_semana}"
    
    def _get_hora_archivo(self) -> str:
        """
        Retorna la hora para el nombre del archivo.
        Formato: HHh-MMm-SSs
        Ejemplo: 08h-30m-15s
        """
        return datetime.now().strftime("%Hh-%Mm-%Ss")
    
    def save(self, frame: np.ndarray, ci: str, resultado: str, 
             tipo_marcado: str = "entrada") -> str:
        """
        Guarda una imagen de captura (solo foto frontal).
        
        Args:
            frame: Imagen BGR (numpy array)
            ci: CI de la persona (con extensión, ej: 1231231-1C)
            resultado: Resultado de la verificación
            tipo_marcado: Tipo de marcado (entrada, salida)
        
        Returns:
            str: Ruta relativa de la imagen guardada o None si no se guarda
        """
        # Si el guardado está deshabilitado
        if not self.enabled:
            return None
        
        # Si es reconocido y no queremos guardar reconocidos
        if resultado == "reconocido" and not self.save_reconocido:
            return None
        
        # Obtener fecha y hora actuales
        fecha_carpeta = self._get_fecha_carpeta()
        hora_archivo = self._get_hora_archivo()
        
        # Crear estructura de carpetas: capturas/{CI}/{fecha_dd-mm-aaaa_Día}/
        dir_path = self.base_dir / str(ci) / fecha_carpeta
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Normalizar resultado para el nombre del archivo
        resultado_nombre = self.RESULTADOS.get(resultado, "error")
        
        # Generar nombre del archivo (sin marcado_id)
        filename = f"{tipo_marcado}_{resultado_nombre}_{hora_archivo}.jpg"
        
        filepath = dir_path / filename
        
        # Guardar imagen
        try:
            cv2.imwrite(str(filepath), frame)
            # Retornar ruta relativa (para guardar en BD)
            ruta_relativa = str(filepath.relative_to(self.base_dir))
            print(f"[CAPTURE] Imagen guardada: {ruta_relativa}", flush=True)
            return ruta_relativa
        except Exception as e:
            print(f"[CAPTURE] Error al guardar imagen: {e}", flush=True)
            return None
    
    def is_enabled(self) -> bool:
        """Retorna si el guardado está habilitado."""
        return self.enabled
    
    def set_enabled(self, enabled: bool):
        """Habilita o deshabilita el guardado."""
        self.enabled = enabled
        print(f"[CAPTURE] Guardado {'habilitado' if enabled else 'deshabilitado'}", flush=True)
    
    def set_save_reconocido(self, save: bool):
        """Configura si guardar verificaciones exitosas."""
        self.save_reconocido = save
        print(f"[CAPTURE] Guardar reconocidos: {'sí' if save else 'no'}", flush=True)


# Instancia global
capture_manager = CaptureManager()