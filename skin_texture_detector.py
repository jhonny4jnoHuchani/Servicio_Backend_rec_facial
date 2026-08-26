"""
skin_texture_detector.py — MODO CALIBRACIÓN.

Calcula las 3 señales (LBP, especular, FFT) e imprime todo en consola,
pero SOLO rechaza usando LBP (que ya vimos que separa bien real/spoof).
Especular y FFT están en modo "solo medir" hasta calibrar sus umbrales
con más pruebas.

Uso: correr /verify o /register varias veces (mezclando fotos reales
y fotos-de-foto) y copiar el bloque [SKIN] de la consola para cada caso,
indicando si era REAL o SPOOF.
"""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern


# ── Umbrales activos (solo LBP por ahora) ──
LBP_ENTROPY_MIN = 3.9
LBP_VARIANCE_MIN = 0.02


class SkinTextureDetector:

    def analyze(self, frame, face_info):
        # ─────────────────────────────────────────────────────────
        # Punto de entrada: recorta el rostro, corre las 3 pruebas,
        # imprime todo, pero decide SOLO con LBP (modo calibración).
        # ─────────────────────────────────────────────────────────
        print("\n[SKIN] ===== Iniciando análisis de textura de piel (MODO CALIBRACIÓN) =====", flush=True)

        x, y, w, h = face_info[:4].astype(int)

        margin_x = int(w * 0.4)
        margin_y = int(h * 0.2)
        x1 = max(0, x - margin_x)
        x2 = min(frame.shape[1], x + w + margin_x)
        y1 = max(0, y - margin_y)
        y2 = min(frame.shape[0], y + h)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            print("[SKIN] Crop vacío, no se puede analizar → rechazado", flush=True)
            return False

        print(f"[SKIN] Crop obtenido: {face_crop.shape[1]}x{face_crop.shape[0]} px", flush=True)

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        # ── Señal 1: LBP (activa) ───────────────────────────────
        avg_entropy, variance = self._lbp_scores(gray)

        # ── Señal 2: brillo especular (solo medir) ──────────────
        specular_ratio = self._specular_ratio(face_crop)

        # ── Señal 3: periodicidad FFT (solo medir) ──────────────
        fft_score = self._fft_periodicity_score(gray)

        print(f"[SKIN] RESUMEN → entropía={avg_entropy:.4f} | varianza={variance:.4f} | "
              f"especular={specular_ratio:.4f} | fft={fft_score:.4f}", flush=True)

        # ── Decisión: SOLO LBP por ahora ────────────────────────
        is_real = avg_entropy > LBP_ENTROPY_MIN and variance > LBP_VARIANCE_MIN
        if not is_real:
            print(f"[SKIN] ❌ RECHAZADO por LBP (entropía={avg_entropy:.4f}, varianza={variance:.4f})", flush=True)
        else:
            print(f"[SKIN] ✅ ACEPTADO por LBP", flush=True)

        return is_real

    def _lbp_scores(self, gray):
        # ─────────────────────────────────────────────────────────
        # Divide el rostro en 3 regiones (mejilla izq, centro,
        # mejilla der), calcula LBP con 2 radios en cada una,
        # y devuelve entropía promedio + varianza entre regiones.
        # ─────────────────────────────────────────────────────────
        print("[SKIN][LBP] Calculando textura por regiones (LBP)...", flush=True)

        h_crop, w_crop = gray.shape
        tercio = w_crop // 3
        regiones = [
            gray[:, :tercio],
            gray[:, tercio:2 * tercio],
            gray[:, 2 * tercio:]
        ]

        scores = []
        for i, region in enumerate(regiones):
            if region.size == 0 or region.shape[0] < 8 or region.shape[1] < 8:
                print(f"[SKIN][LBP] Región {i} descartada (muy chica: {region.shape})", flush=True)
                continue

            lbp1 = local_binary_pattern(region, 8, 1, method='uniform')
            lbp2 = local_binary_pattern(region, 8, 2, method='uniform')

            hist1, _ = np.histogram(lbp1, bins=20, range=(0, 20))
            hist2, _ = np.histogram(lbp2, bins=20, range=(0, 20))

            hist = np.concatenate([hist1, hist2])
            hist = hist / hist.sum()

            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            scores.append(entropy)
            print(f"[SKIN][LBP] Región {i} → entropía={entropy:.4f}", flush=True)

        if not scores:
            print("[SKIN][LBP] No hay regiones válidas → entropía=0, varianza=0", flush=True)
            return 0.0, 0.0

        avg_entropy = float(np.mean(scores))
        variance = float(np.var(scores)) if len(scores) > 1 else 0.0

        print(f"[SKIN][LBP] Resultado → promedio={avg_entropy:.4f} | varianza={variance:.4f}", flush=True)
        return avg_entropy, variance

    def _specular_ratio(self, face_crop):
        # ─────────────────────────────────────────────────────────
        # SOLO MEDICIÓN (no rechaza todavía). Cuenta % de píxeles
        # muy brillantes y poco saturados (reflejo "duro").
        # ─────────────────────────────────────────────────────────
        print("[SKIN][SPECULAR] Analizando reflejo especular (solo medición)...", flush=True)

        hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]

        highlight_mask = (v > 235) & (s < 40)
        ratio = float(np.sum(highlight_mask)) / highlight_mask.size

        print(f"[SKIN][SPECULAR] Píxeles quemados: {np.sum(highlight_mask)}/{highlight_mask.size} "
              f"→ ratio={ratio:.4f}", flush=True)
        return ratio

    def _fft_periodicity_score(self, gray):
        # ─────────────────────────────────────────────────────────
        # SOLO MEDICIÓN (no rechaza todavía). Mide qué tan
        # concentrada está la energía en frecuencias medias-altas
        # (patrones periódicos de impresión/pantalla).
        # ─────────────────────────────────────────────────────────
        print("[SKIN][FFT] Analizando periodicidad de frecuencia (solo medición)...", flush=True)

        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        radius_inner = 8
        radius_outer = max(radius_inner + 1, min(h, w) // 2 - 2)

        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        ring_mask = (dist >= radius_inner) & (dist <= radius_outer)

        ring_energy = magnitude[ring_mask]
        if ring_energy.size == 0:
            print("[SKIN][FFT] Anillo de frecuencias vacío → score=0", flush=True)
            return 0.0

        peak_score = float(np.std(ring_energy) / (np.mean(ring_energy) + 1e-6))

        print(f"[SKIN][FFT] Energía anillo: media={np.mean(ring_energy):.3f}, "
              f"std={np.std(ring_energy):.3f} → score={peak_score:.4f}", flush=True)
        return peak_score