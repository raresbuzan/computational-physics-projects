import numpy as np
import matplotlib.pyplot as plt

# Fast Fourier Transform

def fft_1d(x):
    """
    Implémentation récursive de la FFT (Cooley-Tukey).
    Basé sur le Lemme de Danielson-Lanczos (Section 1.3 du cours).
    x : vecteur d'entrée (taille doit être une puissance de 2)
    """
    N = len(x)

    # Cas de base : si la taille est 1, la FFT est le nombre lui-même
    # (Section 1.3 : "At stage l... period 1")
    if N <= 1:
        return x

    # Division : On sépare les indices pairs (even) et impairs (odd)
    # C'est la séparation f_2j et f_2j+1 du cours (Prop 1.2)
    even = fft_1d(x[0::2])
    odd =  fft_1d(x[1::2])

    # calcul du facteur "Twiddle" : u = exp(-2i * pi * k / N)
    # Note: Le cours utilise u = exp(i...) pour l'inverse ou selon convention,
    # mais la définition standard (Eq 1.3) a un signe moins.
    k = np.arange(N // 2)
    u = np.exp(-2j * np.pi * k / N)

    # Combinaison
    # Fn = F_even + u * F_odd
    # Fn+N/2 = F_even - u * F_odd
    terme_impair = u * odd

    #Concaténation des deux moitiés
    return np.concatenate([even + terme_impair, even - terme_impair])

def fft_2d(image):
    """
    Calcul de la FFT 2D en utilisant la séparabilité.
    FFT 2D = FFT 1D sur les lignes, puis FFT 1D sur les colonnes.
    """
    # Appliquer FFT 1D sur chaque LIGNE
    lignes = np.array([fft_1d(ligne) for ligne in image])

    # Transposer la matrice pour traiter les colonnes comme des lignes
    colonnes = lignes.T

    # Appliquer FFT 1D sur chaque COLONNE (qui sont maintenant des lignes)
    resultat_transpose = np.array([fft_1d(col) for col in colonnes])

    # Retransposer pour revenir à l'orientation originale
    return resultat_transpose.T

def calcul_diffraction(image):
    """
    Calcul la diffraction
    """
    # Appel de NOTRE fonction au lieu de np.fft.fft2
    tf = fft_2d(image)

    # On garde fftshift de numpy car c'est juste du réarrangement d'indices, pas des maths
    tf_centree = np.fft.fftshift(tf)

    # Intensité
    # Note théorique : Selon le cours (Eq 1.4 et Remarks), la DFT inclut un facteur 1/N.
    # Nous l'omettons volontairement ici car nous nous intéressons à la forme de la figure 
    # (proportionnalité I ~ |A|^2) et non à la valeur absolue de l'énergie.
    # L'affichage via imshow normalisera automatiquement les valeurs.
    intensite = np.abs(tf_centree)**2
    return intensite

# Config
N = 512 # Puissance de 2 obligatoire ( ici 2^9 )
centre = N // 2
y, x = np.ogrid[:N, :N] # Création de la grille de coord. x et y pour faciliter les equations de cercles

# 1. Trou carré
image_carre = np.zeros((N, N))
demi = 15
# Slicing ( on découpe le tableau )
image_carre[centre-demi:centre+demi, centre-demi:centre+demi] = 1 # les pixels dans le carré deviennent blanc 
# Calcul
intensite_carre = calcul_diffraction(image_carre)

# 2. Trou circulaire
image_cercle = np.zeros((N, N))
rayon = 15
masque = (x - centre)**2 + (y - centre)**2 <= rayon**2 # equation du cercle
image_cercle[masque] = 1 # On allume les pixels du cercle
# Calcul
intensite_cercle = calcul_diffraction(image_cercle)

# 3. double trou ( trous d'young )
# Exactement le même principe que le cercle simple, mais on en fait juste 2 séparés d'un écart au centre
image_double = np.zeros((N, N))
r_double = 15 # rayon de chaques trous
ecart = 30 # écart avec le centre ( écart entre les 2 trous = ecart*2 )
m1 = (x - (centre - ecart))**2 + (y - centre)**2 <= r_double**2
m2 = (x - (centre + ecart))**2 + (y - centre)**2 <= r_double**2
image_double[m1] = 1
image_double[m2] = 1
# Calcul 
intensite_double = calcul_diffraction(image_double)

# Affichage
plt.figure(figsize=(12, 12)) 

# Ligne 1 : Trou carré
plt.subplot(3, 2, 1)
plt.imshow(image_carre, cmap='afmhot')
plt.title("Ouverture : Carré")
plt.axis('off')

plt.subplot(3, 2, 2)
# On utilise log(I + 1) pour mieux voir les faibles intensités
# On met le +1 pour éviter le log(0)
plt.imshow(np.log(intensite_carre + 1), cmap='afmhot')
plt.title("Diffraction : Croix")
plt.colorbar(label="Intensité (log)")
plt.axis('off')

# Ligne 2 : Trou Circulaire
plt.subplot(3, 2, 3)
plt.imshow(image_cercle, cmap='afmhot')
plt.title("Ouverture : Cercle")
plt.axis('off')

plt.subplot(3, 2, 4)
plt.imshow(np.log(intensite_cercle + 1), cmap='afmhot')
plt.title("Diffraction : Tache d'Airy")
plt.colorbar(label="Intensité (log)")
plt.axis('off')

# Ligne 3 : Double trou
plt.subplot(3, 2, 5)
plt.imshow(image_double, cmap='afmhot')
plt.title("Ouverture : Double Trou")
plt.axis('off')

plt.subplot(3, 2, 6)
plt.imshow(np.log(intensite_double + 1), cmap='afmhot')
plt.title("Diffraction : Interférences de Young")
plt.colorbar(label="Intensité (log)")
plt.axis('off')

plt.tight_layout()
plt.show()