import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

# PARAMETRES

L = 400  # Longueur de la route à bord périodique
DENSITE = 0.3  # pourcentages de la route remplis par des voitures
V_MAX = 5   # Vitesse maximale autorisée
P_FREIN = 0.3 # Pourcentage de chance qu'un conducteur freine sans raison (facteur humain, dépassement, dégats sur la routes, radars)
DT = 150 # On garde en mémoire les 150 dernières étapes pour le graphique "Espace-Temps"

# TRAVAUX (On crée une zone de ralentissement)
DEBUT_TRAVAUX = 280 # Les travaux commencent à la case 280
FIN_TRAVAUX = 300 # Les travaux finissent à la case 300
V_TRAVAUX = 2  # Dans cette zone, la vitesse est limitée à une vitesse fixe

# Initialisation des variables globales
position_repere = -1  # Variable pour stocker la position de la "voiture repère" (-1 = pas encore définie)
# On crée une grande matrice vide (-1 partout) pour stocker l'historique du trafic (image du milieu)
matrice_temps = np.full((DT, L), -1)

# Fonction pour créer la route au tout début
def route(L, densite):
    global position_repere # On va modifier la variable globale position_repere
    
    # On crée une route vide de longueur L remplie de -1 (personne)
    route = np.full(L, -1, dtype=int)
    
    # On calcule combien de voitures on doit placer (Longueur x Densité)
    n_voitures = int(L * densite)
    
    # On choisit 'n_voitures' cases au hasard sur la route pour les placer
    indices = np.random.choice(range(L), n_voitures, replace=False)
    
    # Sur ces cases choisies, on donne une vitesse au hasard entre 0 et V_MAX
    route[indices] = np.random.randint(0, V_MAX + 1, size=n_voitures)
    
    # Si on a mis des voitures, on décide que la première de la liste sera notre "repere"
    if n_voitures > 0:
        position_repere = indices[0]
        
    return route

# LOGIQUE DU TRAFIC

# Cette fonction calcule l'état de la route à l'étape suivante (t+1)
def next_step(route, pos_repere):
    
    # POSITIONS DES VOITURES
    # On récupère les numéros de cases où il y a une voiture (là où ce n'est pas -1)
    pos = np.where(route > -1)[0]
    
    # S'il n'y a pas de voitures, on arrête tout et on renvoie des zéros
    if len(pos) == 0: return route, 0, 0, pos_repere
    
    # On récupère la vitesse actuelle de chaque voiture trouvée
    v = route[pos]

    # CALCUL DES DISTANCES
    # On regarde la voiture suivante. 'roll(-1)' décale le tableau vers la gauche.
    # La voiture i regarde donc la voiture i+1.
    voiture_suivante = np.roll(pos, -1)
    
    # On calcule la distance. Le modulo (% L) gère le fait que la route est un cercle
    # (la dernière voiture regarde la première). Le '-1' car on compte les cases VIDES.
    distance = (voiture_suivante - pos) % L - 1

    # ACCELERATION
    # Tout le monde veut aller plus vite : on ajoute +1 à la vitesse
    # Mais on ne peut pas dépasser V_MAX. 'minimum' garde le plus petit des deux.
    v = np.minimum(v + 1, V_MAX)
    
    # SCENARIO TRAVAUX
    # On crée un masque (vrai/faux) pour savoir qui est dans la zone de travaux
    in_zone = (pos >= DEBUT_TRAVAUX) & (pos <= FIN_TRAVAUX)
    
    # Ceux qui sont dans la zone voient leur vitesse limitée à V_TRAVAUX (2)
    v[in_zone] = np.minimum(v[in_zone], V_TRAVAUX)

    # SECURITE (FREINAGE)
    # Si ma vitesse est plus grande que la distance libre devant moi, je freine.
    # Je prends la vitesse qui correspond à la distance libre pour ne pas taper.
    v = np.minimum(v, distance)

    # FACTEUR HUMAIN (ALEATOIRE)
    # On tire un nombre au hasard entre 0 et 1 pour chaque voiture
    r = np.random.rand(len(v))
    
    # On freine SI (le hasard est < P_FREIN) ET (on roule déjà > 0)
    freiner = (r < P_FREIN) & (v > 0)
    
    # Si la condition est vraie, on enlève 1 à la vitesse (sans descendre sous 0)
    v = np.maximum(v - freiner, 0)

    # DEPLACEMENT 
    # On prépare une nouvelle route vide
    nouvelle_route = np.full(L, -1, dtype=int)
    
    # On calcule les nouvelles positions : (ancienne_pos + vitesse) modulo Longueur
    nouvelles_pos = (pos + v) % L
    
    # On place les voitures avec leur nouvelle vitesse sur la nouvelle route
    nouvelle_route[nouvelles_pos] = v

    # SUIVI DE LA VOITURE REPERE
    # On cherche où est passée notre voiture repère dans le tableau des positions
    idx_repere_local = np.where(pos == pos_repere)[0]
    
    # Si on la trouve, on met à jour sa position globale
    if len(idx_repere_local) > 0:
        nouv_pos_repere = nouvelles_pos[idx_repere_local[0]]
    else:
        nouv_pos_repere = pos_repere # Juste au cas où (sécurité)

    # CALCUL DES STATISTIQUES
    v_moy = np.mean(v) # Vitesse moyenne de toutes les voitures
    flux = DENSITE * v_moy # Flux (combien de voitures passent par seconde)

    # On renvoie tout ce qu'on a calculé pour l'affichage
    return nouvelle_route, v_moy, flux, nouv_pos_repere

# PREPARATION DE L'ANIMATION (PHASE 1)

# On crée la fenêtre graphique (taille 12x10 pouces)
fig = plt.figure(figsize=(12, 10))
# On donne un titre à la fenêtre
fig.canvas.manager.set_window_title('Phase 1 : Observation (Fermez pour voir les stats)')

# On définit une grille pour placer les graphiques (3 lignes, 2 colonnes)
# La ligne du milieu sera 2 fois plus haute que les autres
gs = fig.add_gridspec(3, 2, height_ratios=[0.5, 2, 1], hspace=0.4)

# On crée les zones de dessin (axes) dans la grille
ax_route = fig.add_subplot(gs[0, :])      # En haut, prend toute la largeur
ax_spacetime = fig.add_subplot(gs[1, :])  # Au milieu, prend toute la largeur
ax_vitesse = fig.add_subplot(gs[2, 0])    # En bas à gauche
ax_flux = fig.add_subplot(gs[2, 1])       # En bas à droite

# On définit les couleurs : Blanc(-1), Rouge(0), Orange(1)... jusqu'à Bleu(5) et Noir(10)
colors = ['white', 'red', 'orange', 'yellow', '#ADFF2F', 'green', 'blue', 'black']
# On définit les seuils pour changer de couleur
bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 10.5]
# On crée la palette de couleurs
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# On initialise la route pour de vrai
route = route(L, DENSITE)

# IMAGE 1 : LA ROUTE EN TEMPS REEL
# On affiche la route sous forme d'image (imshow)
img_route = ax_route.imshow(route[np.newaxis, :], aspect='auto', cmap=cmap, norm=norm)
# On dessine la zone grise pour les travaux
ax_route.axvspan(DEBUT_TRAVAUX, FIN_TRAVAUX, color='gray', alpha=0.7)
ax_route.set_title(f"Simulation Temps Réel ( Zone grise = Zone limitée à vitesse = {V_TRAVAUX})" )
ax_route.set_yticks([]) # On enlève les chiffres sur l'axe Y (inutile)
ax_route.set_xticks([]) # On enlève les chiffres sur l'axe X (plus propre)

# ESPACE-TEMPS
# On affiche la matrice d'historique
img_spacetime = ax_spacetime.imshow(matrice_temps, aspect='auto', cmap=cmap, norm=norm, origin='upper')
ax_spacetime.set_title("Diagramme Espace-Temps ( point noir = repère )")
ax_spacetime.set_ylabel("Temps")
ax_spacetime.set_xlabel("Position")
# On ajoute aussi la zone grise de travaux ici
ax_spacetime.axvspan(DEBUT_TRAVAUX, FIN_TRAVAUX, color='gray', alpha=0.7)

# GRAPHIQUES DE STATISTIQUES EN TEMPS REEL
# On crée des listes vides pour stocker les données au fur et à mesure
x_data, y_vitesse, y_flux = [], [], []

# Graphique Vitesse
line_v, = ax_vitesse.plot([], [], 'b-', lw=2) # Ligne bleue
ax_vitesse.set_xlim(0, 300) # L'axe X va de 0 à 300 pas de temps
ax_vitesse.set_ylim(0, V_MAX) # L'axe Y va de 0 à Vitesse Max
ax_vitesse.set_title("Vitesse Moyenne")
ax_vitesse.grid(True) # On met une grille

# Graphique Flux
line_f, = ax_flux.plot([], [], 'm-', lw=2) # Ligne magenta
ax_flux.set_xlim(0, 300)
ax_flux.set_ylim(0, 1.0)
ax_flux.set_title("Débit (Flux)")
ax_flux.grid(True)

# FONCTION D'ANIMATION

def animate(frame):
    # On récupère les variables globales pour les modifier
    global route, position_repere, matrice_temps, x_data, y_vitesse, y_flux
    
    # 1. On fait avancer la simulation d'un pas
    route, v, j, position_repere = next_step(route, position_repere)
    
    # 2. Mise à jour de l'image du milieu (Espace-Temps)
    # On décale tout l'historique vers le haut (on perd la ligne la plus vieille)
    matrice_temps = np.roll(matrice_temps, -1, axis=0)
    
    # On copie la ligne actuelle de la route
    ligne_actuelle = route.copy()
    # Si on a une voiture repere, on force sa couleur à 10 (NOIR) pour la voir
    if position_repere != -1: ligne_actuelle[position_repere] = 10
    
    # On insère cette nouvelle ligne tout en bas de l'historique
    matrice_temps[-1, :] = ligne_actuelle

    # 3. Mise à jour des graphiques du bas
    x_data.append(frame) # On ajoute le numéro de l'image (temps)
    y_vitesse.append(v)  # On ajoute la vitesse moyenne
    y_flux.append(j)     # On ajoute le flux
    
    # Si on dépasse 300 images, on fait défiler le graphique vers la droite
    if frame > 300:
        ax_vitesse.set_xlim(frame - 300, frame)
        ax_flux.set_xlim(frame - 300, frame)
    
    # 4. On met à jour les données visuelles dans la fenêtre
    img_route.set_data(ligne_actuelle[np.newaxis, :])
    img_spacetime.set_data(matrice_temps)
    line_v.set_data(x_data, y_vitesse)
    line_f.set_data(x_data, y_flux)
    
    return img_route, img_spacetime, line_v, line_f

# On lance l'animation (interval=10ms entre chaque image)
ani = animation.FuncAnimation(fig, animate, interval=10, blit=False, cache_frame_data=False)

# On affiche la fenêtre. Le code s'arrête ici tant que la fenêtre est ouverte
plt.show() 

# ANALYSE PHYSIQUE (PHASE 2 )

# Fonction qui fait tourner la simulation "en cachette" (sans dessin) très vite
# pour calculer des moyennes précises sur plein de densités différentes.
def calcul_stat(L_sim, densite_sim, P_FREIN_sim, steps=300, travaux=False):
    # Exactement la même logique que 'route'
    route = np.full(L_sim, -1, dtype=int)
    n = int(L_sim * densite_sim)
    pos = np.random.choice(range(L_sim), n, replace=False)
    route[pos] = np.random.randint(0, V_MAX+1, n)
    
    # Listes pour stocker les mesures stats
    v_moy_samples = []
    v_std_samples = [] 
    
    # On boucle 'steps' fois (ex: 300 tours)
    for i in range(steps):
        # Exactement la même logique que 'next_step'
        pos = np.where(route > -1)[0]
        if len(pos) == 0: break
        v = route[pos]
        dist = (np.roll(pos, -1) - pos) % L_sim - 1
        v = np.minimum(v + 1, V_MAX)
        
        # Gestion des travaux si activés
        if travaux:
            in_zone = (pos >= DEBUT_TRAVAUX) & (pos <= FIN_TRAVAUX)
            v[in_zone] = np.minimum(v[in_zone], V_TRAVAUX)
            
        v = np.minimum(v, dist)
        freiner = (np.random.rand(len(v)) < P_FREIN_sim) & (v > 0)
        v = np.maximum(v - freiner, 0)
        
        new_route = np.full(L_sim, -1, dtype=int)
        new_route[(pos + v) % L_sim] = v
        route = new_route
        
        # MESURE
        # On attend la moitié de la simulation (régime stable) avant de mesurer
        if i > steps // 2:
            v_moy_samples.append(np.mean(v)) # On stocke la vitesse moyenne
            v_std_samples.append(np.std(v))  # On stocke l'écart-type (stabilité)
            
    # On fait la moyenne de toutes les mesures stockées
    v_moy_final = np.mean(v_moy_samples)
    flux_final = densite_sim * v_moy_final
    std_final = np.mean(v_std_samples)
    
    # On renvoie les résultats finaux
    return v_moy_final, flux_final, std_final

# EXPERIENCE A : On teste des densités différentes
# On crée 20 valeurs de densité entre 0.05 et 0.95
densities = np.linspace(0.05, 0.95, 20)
# Listes pour stocker les résultats
flux_res = []
vitesse_res = []
std_res = [] 

# On boucle sur chaque densité
for d in densities:
    # On lance la simulation rapide
    v, f, s = calcul_stat(L, d, P_FREIN, travaux=False) 
    # On ajoute les résultats aux listes
    flux_res.append(f)
    vitesse_res.append(v)
    std_res.append(s)

# EXPÉRIENCE B : On teste l'impact du freinage aléatoire
# On crée 20 valeurs de probabilité entre 0 (robot) et 1 (panique)
probas_frein = np.linspace(0, 1.0, 20)
vitesse_vs_proba = []

densite_fixe = 0.3 # Densité fixée à 0.3 (régime congestionné, cohérent avec le rapport)
# On boucle sur chaque probabilité
for p in probas_frein:
    v, _, _ = calcul_stat(L, densite_fixe, p, travaux=False)
    vitesse_vs_proba.append(v)

# AFFICHAGE DES RÉSULTATS (PHASE 3)

# On prépare une nouvelle figure avec 4 graphiques (2x2)
fig2, axs = plt.subplots(2, 2, figsize=(14, 10))
fig2.canvas.manager.set_window_title('Phase 2 : Analyse Physique Complète')

# On met un peu d'espace entre les graphiques (esthétisme)
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# GRAPHIQUE 1: FLUX
axs[0, 0].plot(densities, flux_res, 'o-', color='purple')
axs[0, 0].set_title("1. Diagramme Fondamental (Flux)")
axs[0, 0].set_xlabel(r"Densité ($\rho$)")
axs[0, 0].set_ylabel(r"Flux ($J$)")
axs[0, 0].grid(True) # Grille

# GRAPHIQUE 2 : VITESSE
axs[0, 1].plot(densities, vitesse_res, 'o-', color='blue')
axs[0, 1].set_title("2. Vitesse Moyenne vs Densité")
axs[0, 1].set_xlabel(r"Densité ($\rho$)")
axs[0, 1].set_ylabel("Vitesse")
axs[0, 1].grid(True)

# GRAPHIQUE 3 : INSTABILITÉ
axs[1, 0].plot(densities, std_res, 'o-', color='red', lw=2)
axs[1, 0].set_title(r"3. Instabilité ($\sigma_v$)")
axs[1, 0].set_xlabel(r"Densité ($\rho$)")
axs[1, 0].set_ylabel("Écart-type")
axs[1, 0].grid(True)

# GRAPHIQUE 4 : FACTEUR HUMAIN
axs[1, 1].plot(probas_frein, vitesse_vs_proba, 's-', color='green', lw=2)
axs[1, 1].set_title(r"4. Impact du Facteur Humain")
axs[1, 1].set_xlabel(r"Probabilité de freinage ($P_{frein}$)")
axs[1, 1].set_ylabel("Vitesse Moyenne")
axs[1, 1].grid(True)
# Petit encadré pour dire que la densité est de 0.3
axs[1, 1].text(0.1, 0.2, f"Densité fixée à {densite_fixe}", transform=axs[1, 1].transAxes, bbox=dict(facecolor='white', alpha=0.8))

print(">>> TERMINÉ.")
# On affiche la fenêtre finale
plt.show()