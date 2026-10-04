import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

# Paramètres de la simulation (en unités réduites LJ : m=1, kB=1)
N = 30
boxsize = 10.0
dt = 0.01
epsilon = 1.0
sigma = 1.0
temp_ini = 0.5 

np.random.seed(42) # Fixe la graine pour la reproductibilité (pour comparer l'effet des conditions initiales)

# Initialisation aléatoire des positions et tirage des vitesses selon une distribution gaussienne
positions = np.random.rand(N, 2) * boxsize
vitesses = np.random.normal(0, np.sqrt(temp_ini), (N, 2))
accelerations = np.zeros((N, 2))

# Variables pour l'historique (tracé des énergies et de la pression)
hist = 200
ep_hist, ec_hist, et_hist = [0]*hist, [0]*hist, [0]*hist
P_hist = [0.0]*hist
x_list = list(range(hist))

v_bins = np.linspace(0, 4, 15)
v_centers = (v_bins[:-1] + v_bins[1:]) / 2

# On stocke les trajectoires de quelques particules pour la visualisation
nbr_traj = 5
long_traj = 400
traj_history = np.zeros((nbr_traj, long_traj, 2))
for i in range(nbr_traj):
    traj_history[i, :, :] = positions[i]

# Configuration de l'affichage Matplotlib
plt.rcParams['toolbar'] = 'None' # Sert à enlever le menu en bas ( purement visuel )
fig = plt.figure(figsize=(11, 8))
gs = fig.add_gridspec(4, 2, width_ratios=[1.2, 1], height_ratios=[0.5, 1, 1, 0.1])

ax_sim = fig.add_subplot(gs[0:3, 0])
ax_sim.set_xlim(0, boxsize); ax_sim.set_ylim(0, boxsize)
ax_sim.set_aspect('equal')
ax_sim.set_xticks([]); ax_sim.set_yticks([])
ax_sim.set_title('Gaz Lennard-Jones 2D')

# Le scatter plot colore les particules selon la norme de leur vitesse
scat = ax_sim.scatter(positions[:, 0], positions[:, 1], s=150, c=np.linalg.norm(vitesses, axis=1), cmap='coolwarm', edgecolors='black', vmin=0, vmax=3)
lines_traj = [ax_sim.plot([], [], '-', linewidth=1.5, alpha=0.7)[0] for _ in range(nbr_traj)]

ax_text = fig.add_subplot(gs[0, 1])
ax_text.axis('off')
time_text = ax_text.text(0.0, 0.5, '', fontsize=11, family='monospace', verticalalignment='center')

ax_graph = fig.add_subplot(gs[1, 1])
ax_graph.set_xlim(0, hist)
ax_graph.set_title("Energies ( sur les 200 derniers pas de temps )", fontsize=10)
ax_graph.grid(True, linestyle='--', alpha=0.5)
line_ec, = ax_graph.plot([], [], color='red', label='Ec', linewidth=1.5)
line_ep, = ax_graph.plot([], [], color='green', label='Ep', linewidth=1.5)
line_et, = ax_graph.plot([], [], color='black', linestyle=':', label='Etot', linewidth=1)
ax_graph.legend(loc='upper right', fontsize=8)

ax_hist = fig.add_subplot(gs[2, 1])
ax_hist.set_xlim(0, 4)
ax_hist.set_ylim(0, 1.5)
ax_hist.set_title("Vitesses ( Maxwell-Boltzmann )", fontsize=10)
ax_hist.grid(True, linestyle='--', alpha=0.5)
line_hist, = ax_hist.plot([], [], drawstyle='steps-mid', color='blue', label='Simu')
line_mb, = ax_hist.plot([], [], 'r--', label='Théorie')
ax_hist.legend(loc='upper right', fontsize=8)

ax_slider = fig.add_subplot(gs[3, :])
slider_temp = Slider(ax_slider, 'Temp', 0.0, 5.0, valinit=temp_ini, color='orange')

# Slider de température pour la faire varier pendant la simu
def update_slider(val):
    global temp_ini
    temp_ini = slider_temp.val
slider_temp.on_changed(update_slider)

# Boucle d'intégration principale
def update(frame):
    # On importe les variables en global pour pouvoir les modifier à chaque frame
    global positions, vitesses, accelerations, ep_hist, ec_hist, et_hist, P_hist

    # Algorithme de Verlet-vitesse (partie 1).
    vitesses += 0.5 * accelerations * dt
    positions += vitesses * dt

    # Conditions aux limites : Murs réfléchissants durs
    for i in range(N):
        if positions[i, 0] < 0: positions[i, 0] = -positions[i, 0]; vitesses[i, 0] *= -1
        elif positions[i, 0] > boxsize: positions[i, 0] = 2*boxsize - positions[i, 0]; vitesses[i, 0] *= -1
        
        if positions[i, 1] < 0: positions[i, 1] = -positions[i, 1]; vitesses[i, 1] *= -1
        elif positions[i, 1] > boxsize: positions[i, 1] = 2*boxsize - positions[i, 1]; vitesses[i, 1] *= -1

    accelerations.fill(0)
    Ep = 0
    viriel = 0

    # Calcul des forces d'interaction (Potentiel de Lennard-Jones)
    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist_sq = dx**2 + dy**2
            r = np.sqrt(dist_sq)
            
            # sécurité pour éviter l'explosion numérique si les particules se chevauchent
            if r < 0.8: 
                r = 0.8; dist_sq = r*r
            
            # Optimisation des calculs : on évite les puissances gourmandes en calculant d'abord r^2
            sr2 = (sigma**2) / dist_sq 
            sr6 = sr2 ** 3
            sr12 = sr6 ** 2

            # facteur_force correspond à la norme de la force divisée par la distance r : |F|/r = - (1/r) * dV/dr
            # Cela permet d'obtenir Fx et Fy via une simple multiplication par dx et dy
            # Dérivation du potentiel V(r) pour obtenir les forces
            facteur_force = (24 * epsilon / dist_sq) * (2 * sr12 - sr6)
            fx = facteur_force * dx
            fy = facteur_force * dy
            
            # Principe d'action-réaction (3ème loi de Newton) pour diviser le nombre de boucles par 2
            accelerations[i, 0] += fx; accelerations[i, 1] += fy
            accelerations[j, 0] -= fx; accelerations[j, 1] -= fy
            
            Ep += 4 * epsilon * (sr12 - sr6)
            # Le terme du viriel est le produit scalaire r_ij.F_ij
            # Comme F_ij = facteur_force * r_ij, alors r_ij.F_ij = facteur_force * (r_ij)^2 = facteur_force * dist_sq
            viriel += facteur_force * dist_sq

    # Verlet-vitesse (partie 2)
    vitesses += 0.5 * accelerations * dt

    # Grandeurs thermodynamiques
    Ec = 0.5 * np.sum(vitesses**2)
    temp_actu = Ec / N # Température cinétique instantanée (théorème d'équipartition)
    # Théorème d'équipartition de l'énergie en 2D : Ec = N * (2/2) * kB * T
    # En unités réduites (kB=1), cela se simplifie directement en T = Ec / N
    
    # Calcul de la pression instantanée via le théorème du viriel en 2D (P = (Ec + W) / V)
    P_inst = (Ec + 0.5 * viriel) / (boxsize ** 2)
    P_hist.pop(0); P_hist.append(P_inst)
    P_moy = np.mean(P_hist)

    # Thermostat (type Berendsen)
    if temp_actu > 0:
        scale = np.sqrt(temp_ini / temp_actu)
        # Le facteur 0.1 (tau) sert d'amortisseur pour ne pas brusquer le système
        vitesses *= 1.0 + 0.1 * (scale - 1.0)

    # Mise à jour visuelle des graphes
    Et = Ec + Ep
    ep_hist.pop(0); ep_hist.append(Ep)
    ec_hist.pop(0); ec_hist.append(Ec)
    et_hist.pop(0); et_hist.append(Et)
    
    norme_vit = np.linalg.norm(vitesses, axis=1)
    
    scat.set_offsets(positions)
    scat.set_array(norme_vit)
    
    line_ep.set_data(x_list, ep_hist)
    line_ec.set_data(x_list, ec_hist)
    line_et.set_data(x_list, et_hist)
    ax_graph.set_ylim(min(min(ep_hist), min(ec_hist)) - 5, max(max(ep_hist), max(ec_hist)) + 5)

    # density=True est crucial ici, il normalise l'aire de l'histogramme à 1 
    # pour qu'on puisse le superposer à la fonction de probabilité théorique
    counts, _ = np.histogram(norme_vit, bins=v_bins, density=True)
    line_hist.set_data(v_centers, counts)
    
    # Superposition de la distribution de Maxwell-Boltzmann théorique en 2D pour vérifier la thermalisation
    if temp_actu > 0.01:
        f_v_theorie = (v_bins / temp_actu) * np.exp(-(v_bins**2) / (2 * temp_actu))
        line_mb.set_data(v_bins, f_v_theorie)
    else:
        line_mb.set_data([], [])

    for i in range(nbr_traj):
        traj_history[i, 1:] = traj_history[i, :-1]
        traj_history[i, 0] = positions[i]
        lines_traj[i].set_data(traj_history[i, :, 0], traj_history[i, :, 1])

    info = (f"Temp Cible   : {temp_ini:.2f}\n"
            f"Temp Réelle  : {temp_actu:.2f}\n"
            f"P inst       : {P_inst:.2f}\n"
            f"P moy        : {P_moy:.2f}\n\n"
            f"Ec   : {Ec:.1f}\n"
            f"Ep   : {Ep:.1f}\n"
            f"Etot : {Et:.1f}")
    time_text.set_text(info)
    
    return scat, time_text, line_ep, line_ec, line_et, line_hist, line_mb, *lines_traj

fig.tight_layout()
ani = animation.FuncAnimation(fig, update, frames=200, interval=20, blit=False)
plt.show()