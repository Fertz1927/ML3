import numpy as np
import cv2
from skimage import io, color 
import matplotlib
matplotlib.use('Agg') # Configura Matplotlib para trabajar en modo sin pantalla (headless)
import matplotlib.pyplot as plt


img_original = io.imread('gengar.png')

if len(img_original.shape) == 3: # ¿es de 3D?
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)
else:
    img_gray = img_original

# Redimenzionar si esnnecesario:
if img_gray.shape != (1080, 1920):
    img_gray = cv2.resize(img_gray, (1920, 1080))

U, S, Vt = np.linalg.svd(img_gray, full_matrices=False) # Calcular la Descomposición en Valores Singulares (SVD. U tendrá forma (1080, 1080), S un vector de 1080 valores, Vt forma (1080, 1920)
#full_matrices=False, elimina datos inútiles de los calculos, al dejar S como vector 1D y Vt del mismo tamaño que A y no 1920x1920.

D = np.sum(S) #  suma total D y las proporciones pi
p_i = S / D

P_N = np.cumsum(p_i) # suma acumulada PN para encontrar N (cobertura del 95% = 0.95)

N = np.argmax(P_N >= 0.95) + 1 # busca el primer índice donde la suma acumulada alcance o supere 0.95 (+1 por índice 0)

print(f"Suma total de valores singulares (D): {D:.2f}")
print(f"Número de términos N necesarios para conservar el 95% de la información: {N}")

U_N = U[:, :N]              # Mantiene las primeras N columnas
S_N = S[:N]                 # Mantiene los primeros N valores singulares
Vt_N = Vt[:N, :]            # Mantiene las primeras N filas
# Recortar las matrices y los valores singulares a N términos

print(f"Forma de U recortada: {U_N.shape}")
print(f"Forma de S recortada: {S_N.shape}")
print(f"Forma de Vt recortada: {Vt_N.shape}")

img_reconstruida = np.dot(U_N, np.dot(np.diag(S_N), Vt_N))

img_reconstruida = np.clip(img_reconstruida, 0, 255).astype(np.uint8) # Convertir los valores a un rango válido de imagen de 8 bits (0 - 255)

cv2.imwrite('imagen_reconstruida.png', img_reconstruida)

plt.figure(figsize=(10, 6))
plt.imshow(img_reconstruida, cmap='gray')
plt.title(f"Imagen Reconstruida (N={N})")
plt.axis('off')
plt.savefig('grafico_reconstruido.png', bbox_inches='tight')
plt.close()

print("¡Imagen reconstruida guardada exitosamente como 'imagen_reconstruida.png'!")
