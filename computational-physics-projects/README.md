# Computational Physics Projects

Four numerical physics projects carried out during my Master 1 in Physics (computational track) at CY Cergy Paris Université, 2025–2026.
All projects were done in pairs with **Amaury Mulloni**. Code is written in Python (NumPy, Matplotlib); each folder contains the source code, the original report (in French) and figures.

| Project | Physics | Numerical methods |
|---|---|---|
| [Time-dependent Schrödinger equation](schrodinger-1d/) | Quantum tunneling through a potential barrier | Finite differences, explicit/implicit Euler, Crank–Nicolson, leapfrog, spectral reference, von Neumann stability |
| [Traffic flow (Nagel–Schreckenberg)](traffic-nagel-schreckenberg/) | Phantom traffic jams, phase transition, fundamental diagram | Stochastic cellular automaton, NumPy vectorization, periodic boundaries |
| [Molecular dynamics of a 2D Lennard-Jones gas](molecular-dynamics-lj/) | Condensation vs gas phase, Maxwell–Boltzmann statistics, virial pressure | Velocity-Verlet integrator, Berendsen thermostat |
| [Fraunhofer diffraction with a hand-written FFT](fft-diffraction/) | Diffraction by square, circular and double apertures | Recursive radix-2 Cooley–Tukey FFT, 2D separability |

## Run

```bash
pip install -r requirements.txt
python schrodinger-1d/schrodinger_1d.py
```

Each script opens an interactive Matplotlib window.

## Authors

Amaury Mulloni and Rares Buzan — CY Cergy Paris Université.
