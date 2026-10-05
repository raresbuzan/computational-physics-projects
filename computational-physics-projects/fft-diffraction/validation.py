"""Quantitative checks of the hand-written FFT (fft_diffraction.py).

1. Accuracy against numpy.fft.fft2
2. Execution time against numpy.fft.fft2
3. First minimum of the square-aperture pattern (sinc^2)
4. First dark ring of the circular-aperture pattern (Airy)
5. Fringe spacing of the double aperture (Young)
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None  # the main script ends with plt.show()

import fft_diffraction as fd

N, c = fd.N, fd.centre

# 1. Accuracy
err = np.max(np.abs(fd.fft_2d(fd.image_cercle) - np.fft.fft2(fd.image_cercle)))
ref = np.max(np.abs(np.fft.fft2(fd.image_cercle)))
print(f"Max |fft_2d - np.fft.fft2| / max|F| : {err/ref:.1e}")

# 2. Timing
t0 = time.perf_counter(); fd.fft_2d(fd.image_cercle); t1 = time.perf_counter()
np.fft.fft2(fd.image_cercle); t2 = time.perf_counter()
print(f"Time fft_2d (recursive, pure Python): {t1-t0:.2f} s | np.fft.fft2: {(t2-t1)*1e3:.1f} ms")

def first_min(profile):
    """Index distance from the centre to the first local minimum."""
    for k in range(1, len(profile) - 1):
        if profile[k] < profile[k-1] and profile[k] <= profile[k+1]:
            return k
    return None

# 3. Square: theory N / a, a = 2*demi
a = 2 * fd.demi
d_sq = first_min(fd.intensite_carre[c, c:])
print(f"Square first minimum : {d_sq} px (theory N/a = {N/a:.2f} px)")

# 4. Airy: first dark ring at 1.22 N / (2R)
d_airy = first_min(fd.intensite_cercle[c, c:])
print(f"Airy first dark ring : {d_airy} px (theory 1.22 N/(2R) = {1.22*N/(2*fd.rayon):.2f} px)")

# 5. Young fringes: spacing N / D along the horizontal axis, D = 2*ecart
D = 2 * fd.ecart
prof = fd.intensite_double[c, :]
mins = [k for k in range(c - 40, c + 41) if prof[k] < prof[k-1] and prof[k] <= prof[k+1]]
print(f"Young fringe spacing : {np.mean(np.diff(mins)):.2f} px (theory N/D = {N/D:.2f} px)")
