import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Paramètres spatiaux
# On prend L=10 pour être large et éviter que l'onde tape les bords (conditions de Dirichlet)
L = 10.0            
N = 250             
x = np.linspace(-L, L, N)
dx = x[1] - x[0]

# Paramètres temporels
# attention : dt fixé à 0.001 à cause de la méthode Leapfrog. 
# Si on passe à 0.003, la condition CFL n'est plus respectée et la norme explose.
dt = 0.001         

# potentiel V(x) 
# on place une barrière de hauteur 30 entre x=1 et x=2
V0 = 30.0 # Potentiel de la barrière
V = np.zeros(N)
for i in range(N):
    if 1.0 < x[i] < 2.0:
        V[i] = V0

# état initial
# paquet d'onde gaussien classique qui fonce vers la barrière
x0 = -4.0
sigma = 0.8
k0 = 7.0 
psi_initial = np.exp(-0.5 * ((x - x0) / sigma)**2) * np.exp(1j * k0 * x)

# On normalise pour que l'intégrale vaille 1 (très important pour le suivi de la probabilité)
norme = np.sqrt(np.sum(np.abs(psi_initial)**2) * dx)
psi_initial /= norme

# construction de l'Hamiltonien (H)
# matrice tridiagonale. Le premier et le dernier point restent à 0 (bords).
H = np.zeros((N, N), dtype=complex)
for i in range(N):
    H[i, i] = 1.0 / dx**2 + V[i] 
    if i > 0:   H[i, i-1] = -0.5 / dx**2
    if i < N-1: H[i, i+1] = -0.5 / dx**2

I = np.eye(N, dtype=complex) # matrice Identité

# Pré-calculs pour optimiser l'affichage
# Si on inverse les matrices dans la boucle d'animation, ça ralenti beaucoup trop.
M_gauche = I + 1j * (dt/2) * H
M_droite = I - 1j * (dt/2) * H
inv_imp = np.linalg.inv(I + 1j * dt * H)
inv_M_gauche = np.linalg.inv(M_gauche)

# Méthode Spectrale (pour comparer avec une solution plus précise)
energies, modes_propres = np.linalg.eigh(H)
coefficients = np.dot(np.conj(modes_propres.T), psi_initial)

#Initialisation des états pour la simulation
psi_exp = np.copy(psi_initial)
psi_imp = np.copy(psi_initial)
psi_cn  = np.copy(psi_initial)

# Pour Leapfrog il nous faut deux pas de temps initiaux. On triche en utilisant Crank-Nicolson pour le premier.
psi_lf_prev = np.copy(psi_initial)
psi_lf_curr = np.dot(inv_M_gauche, np.dot(M_droite, psi_initial))

temps_liste = [0.0]
prob_exp, prob_imp, prob_cn, prob_lf = [1.0], [1.0], [1.0], [1.0]

# graphiques
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# graphe de gauche : L'effet tunnel
ax1.set_xlim(-L, L)
ax1.set_ylim(0, 1.6)
ax1.set_title("Observation de l'Effet Tunnel")
ax1.set_ylabel(r"Densité de probabilité $|\psi|^2$")
ax1.plot(x, V/V0, color='gray', alpha=0.3, label="Barrière")

line_cn, = ax1.plot(x, np.abs(psi_cn)**2, color='blue', lw=2, alpha=0.8, label="Crank-Nicolson")
line_lf, = ax1.plot(x, np.abs(psi_lf_curr)**2, color='cyan', ls='-.', alpha=0.8, label="Leapfrog")
line_sp, = ax1.plot(x, np.abs(psi_initial)**2, color='magenta', ls=':', alpha=0.8, label="Spectrale (Précise)")
ax1.legend(loc="upper left")

# graphe de droite : Suivi de la norme
ax2.set_xlim(0, 1.0)
ax2.set_ylim(0, 3.0)
ax2.set_title("Conservation de la probabilité")
ax2.set_xlabel("Temps t")

line_p_exp, = ax2.plot(temps_liste, prob_exp, color='red', label="Euler Explicite (Explose)")
line_p_imp, = ax2.plot(temps_liste, prob_imp, color='orange', label="Euler Implicite (Dissipe)")
line_p_cn,  = ax2.plot(temps_liste, prob_cn, color='green', label="Crank-Nicolson (Unitaire)")
line_p_lf,  = ax2.plot(temps_liste, prob_lf, color='cyan', label="Leapfrog", ls='--')
ax2.legend(loc="upper left")

t_global = 0.0

# animation
def update(frame):
    global psi_exp, psi_imp, psi_cn, psi_lf_prev, psi_lf_curr, t_global
    
    # On fait 25 itérations sans mettre à jour le graph pour accélérer l'animation 
    for _ in range(25): 
        t_global += dt 
        
        # Euler Explicite 
        if prob_exp[-1] < 10.0:  #On bloque le calcul si ça part trop loin pour éviter un crash complet
            psi_exp = np.dot((I - 1j * dt * H), psi_exp)
            
        # Euler Implicite
        psi_imp = np.dot(inv_imp, psi_imp)
        
        # Crank-Nicolson
        psi_cn = np.dot(inv_M_gauche, np.dot(M_droite, psi_cn))
        
        # Leapfrog 
        if prob_lf[-1] < 10.0: 
            psi_lf_next = psi_lf_prev - 2j * dt * np.dot(H, psi_lf_curr)
            psi_lf_prev = np.copy(psi_lf_curr)
            psi_lf_curr = np.copy(psi_lf_next)
        
        # Mise à jour des probas
        temps_liste.append(t_global)
        prob_exp.append(np.sum(np.abs(psi_exp)**2) * dx if prob_exp[-1] < 10.0 else prob_exp[-1])
        prob_imp.append(np.sum(np.abs(psi_imp)**2) * dx)
        prob_cn.append(np.sum(np.abs(psi_cn)**2) * dx)
        prob_lf.append(np.sum(np.abs(psi_lf_curr)**2) * dx if prob_lf[-1] < 10.0 else prob_lf[-1])

    # Evolution de la méthode spéctrale
    psi_sp = np.dot(modes_propres, coefficients * np.exp(-1j * energies * t_global))

    # Mise à jour graphique
    line_cn.set_ydata(np.abs(psi_cn)**2)
    line_lf.set_ydata(np.abs(psi_lf_curr)**2)
    line_sp.set_ydata(np.abs(psi_sp)**2)
    
    line_p_exp.set_data(temps_liste, prob_exp)
    line_p_imp.set_data(temps_liste, prob_imp)
    line_p_cn.set_data(temps_liste, prob_cn)
    line_p_lf.set_data(temps_liste, prob_lf)
    
    # Défilement de l'axe X pour le graphe de droite
    if t_global > ax2.get_xlim()[1]:
        ax2.set_xlim(0, t_global + 0.5)
        
    # calcul pour l'affichage du pourcentage qui a traversé la barrière (x > 2.0)
    transmission = np.sum(np.abs(psi_cn[x > 2.0])**2) * dx
    ax1.set_xlabel(f"Position x  |  Transmission : {transmission*100:.2f} %")
    
    return line_cn, line_lf, line_sp, line_p_exp, line_p_imp, line_p_cn, line_p_lf

ani = animation.FuncAnimation(fig, update, frames=200, interval=30, blit=False)
plt.tight_layout()
plt.show()