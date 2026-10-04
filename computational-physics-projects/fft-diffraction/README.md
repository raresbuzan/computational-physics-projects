# Fraunhofer diffraction with a hand-written FFT

In the far field, the diffracted amplitude is the Fourier transform of the aperture. This project computes it with a recursive radix-2 Cooley–Tukey FFT written from scratch (no `np.fft.fft2`), on a 512 × 512 grid.

![Diffraction patterns](figures/diffraction_patterns.png)

## Method
- 1D FFT by the Danielson–Lanczos lemma (even/odd split, twiddle factors), O(N log N).
- 2D FFT by separability (rows, then columns).
- Intensity |A|² shown on a logarithmic scale to reveal secondary maxima.

## Results
Square aperture: sinc² cross pattern. Circular aperture: Airy pattern. Double aperture: Airy envelope modulated by Young fringes.

## Known limitations
- No quantitative check yet against `np.fft.fft2`, against the analytical first Airy zero, or of execution time.

## Run
`python fft_diffraction.py` — report (French): [report_fr.pdf](report_fr.pdf). Joint work with Amaury Mulloni.
