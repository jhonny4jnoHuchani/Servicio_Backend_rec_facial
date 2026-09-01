import numpy as np


class GestureDetector:
    """
    Compara landmarks faciales entre una foto frontal y una foto con
    gesto para verificar que el gesto solicitado fue realizado.

    Índices de landmarks (YuNet):
      0,1: x,y rostro     2,3: w,h rostro
      4,5: ojo derecho    6,7: ojo izquierdo
      8,9: nariz          10,11: boca derecha   12,13: boca izquierda
      14: score de detección
    """

    OJO_D_X, OJO_D_Y = 4, 5
    OJO_I_X, OJO_I_Y = 6, 7
    NARIZ_X, NARIZ_Y = 8, 9
    BOCA_D_X, BOCA_D_Y = 10, 11
    BOCA_I_X, BOCA_I_Y = 12, 13
    SCORE = 14

    # --- Umbrales calibrados con datos reales (ver logs 27-28/08/2026) ---
    MIN_DETECTION_SCORE = 0.5   # confianza mínima de YuNet en ambas fotos
    UMBRAL_VERTICAL = 0.025       # bajado de 0.12: el máximo delta real observado
                                  # en "arriba" fue ~0.09, muchos casos entre 0.03-0.07
    UMBRAL_ASIMETRIA = 0.12      # se mantiene; una vez corregido el signo (ver abajo),
                                  # los deltas reales de "derecha"/"izquierda" estuvieron
                                  # entre 0.22 y 0.56, muy por encima de este umbral
    UMBRAL_SONRISA = 0.03        # bajado de 0.15: deltas reales entre 0.07-0.11

    def verify(self, frontal_face, gesto_face, gesto_solicitado):
        if frontal_face is None or gesto_face is None:
            return False

        f_w, f_h = frontal_face[2], frontal_face[3]
        g_w, g_h = gesto_face[2], gesto_face[3]
        if f_w <= 0 or f_h <= 0 or g_w <= 0 or g_h <= 0:
            return False

        # Landmarks poco confiables (blur, ángulo extremo, etc.) -> no decidir
        if frontal_face[self.SCORE] < self.MIN_DETECTION_SCORE or \
        gesto_face[self.SCORE] < self.MIN_DETECTION_SCORE:
            print(f"[GESTO-DEBUG] Score bajo: frontal={frontal_face[self.SCORE]:.4f} gesto={gesto_face[self.SCORE]:.4f}", flush=True)
            return False

        # DEBUG temporal
        print(f"[GESTO-DEBUG] solicitado={gesto_solicitado}", flush=True)
        print(f"[GESTO-DEBUG] t_vertical frontal={self._t_vertical(frontal_face):.4f} gesto={self._t_vertical(gesto_face):.4f}", flush=True)
        print(f"[GESTO-DEBUG] asimetria_h frontal={self._asimetria_horizontal(frontal_face):.4f} gesto={self._asimetria_horizontal(gesto_face):.4f}", flush=True)
        print(f"[GESTO-DEBUG] ancho_boca frontal={self._ancho_boca_norm(frontal_face):.4f} gesto={self._ancho_boca_norm(gesto_face):.4f}", flush=True)
        # FIN DEBUG

        # NOTA IMPORTANTE (calibración 27-28/08/2026):
        # Con datos reales (cámara sin espejo confirmado en APK y backend),
        # se encontró que "izquierda" y "derecha" tenían el signo invertido:
        # al girar a la derecha, asimetria_h SIEMPRE bajaba (5/5 pruebas),
        # cuando el código anterior esperaba que subiera, y viceversa.
        # Se invirtieron los signos de dirección abajo. "abajo" mostró el
        # mismo patrón sospechoso con menos muestras (2/3); se invirtió
        # también pero debe seguir validándose con más pruebas.
        checks = {
            "arriba": lambda: self._vertical(frontal_face, gesto_face, direccion=-1),
            "abajo": lambda: self._vertical(frontal_face, gesto_face, direccion=1),   # antes +1
            "izquierda": lambda: self._horizontal(frontal_face, gesto_face, direccion=1),   # antes -1
            "derecha": lambda: self._horizontal(frontal_face, gesto_face, direccion=-1),    # antes +1
            "sonrisa": lambda: self._sonrisa(frontal_face, gesto_face),
        }
        fn = checks.get(gesto_solicitado)
        return fn() if fn else False

    # -- Utilidades de normalización -----------------------------------

    def _dist(self, ax, ay, bx, by):
        return float(np.hypot(ax - bx, ay - by))

    def _t_vertical(self, face):
        """
        Posición relativa de la nariz entre la línea de ojos y la altura
        del rostro, normalizada por el alto del bbox (independiente de
        la distancia a la cámara). Sube/baja con el pitch de la cabeza.
        """
        ojo_y = (face[self.OJO_D_Y] + face[self.OJO_I_Y]) / 2
        h = face[3]
        if h == 0:
            return 0.0
        return (face[self.NARIZ_Y] - ojo_y) / h

    def _asimetria_horizontal(self, face):
        """
        Razón entre distancia nariz-ojoDerecho y nariz-ojoIzquierdo,
        normalizada por el ancho del rostro. ≈1 de frente; se aleja de 1
        al girar la cabeza (yaw), y el SIGNO del desvío indica direcció.n.
        """
        w = face[2]
        if w == 0:
            return 1.0
        d_der = self._dist(face[self.NARIZ_X], face[self.NARIZ_Y],
                            face[self.OJO_D_X], face[self.OJO_D_Y]) / w
        d_izq = self._dist(face[self.NARIZ_X], face[self.NARIZ_Y],
                            face[self.OJO_I_X], face[self.OJO_I_Y]) / w
        if d_izq == 0:
            return 1.0
        return d_der / d_izq

    def _ancho_boca_norm(self, face):
        w = face[2]
        if w == 0:
            return 0.0
        return self._dist(face[self.BOCA_D_X], face[self.BOCA_D_Y],
                           face[self.BOCA_I_X], face[self.BOCA_I_Y]) / w

    # -- Verificaciones por gesto ----------------------------------------

    def _vertical(self, f, g, direccion):
        # direccion: -1 = "más arriba" (t_vertical baja), +1 = "más abajo" (t_vertical sube)
        # NOTA: validado empíricamente el 27-28/08/2026 para "arriba".
        # "abajo" se invirtió por patrón similar pero con menos muestras (2/3);
        # revisar los próximos logs para confirmar o revertir.
        delta = (self._t_vertical(g) - self._t_vertical(f)) * direccion
        return delta > self.UMBRAL_VERTICAL

    def _horizontal(self, f, g, direccion):
        # direccion: signo validado empíricamente el 27-28/08/2026 con 5/5
        # pruebas reales para "derecha" (asimetria_h bajaba consistentemente
        # al girar a la derecha) e izquierda con patrón inverso.
        delta = (self._asimetria_horizontal(g) - self._asimetria_horizontal(f)) * direccion
        return delta > self.UMBRAL_ASIMETRIA

    def _sonrisa(self, f, g):
        """
        Verifica que la boca se ENSANCHE en la foto de gesto.
        Una sonrisa real aumenta el ancho de la boca.
        """
        boca_f = self._ancho_boca_norm(f)
        boca_g = self._ancho_boca_norm(g)
        
        if boca_f == 0:
            return False
        
        # Cambio relativo (positivo = ensancha, negativo = estrecha)
        cambio = (boca_g - boca_f) / boca_f
        
        # DEBUG
        print(f"[GESTO-DEBUG] sonrisa: boca_f={boca_f:.4f} boca_g={boca_g:.4f} cambio={cambio:.4f} umbral={self.UMBRAL_SONRISA}", flush=True)
        
        # Solo aceptar si la boca se ENSANCHA más que el umbral
        return cambio > self.UMBRAL_SONRISA