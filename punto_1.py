import numpy as np
import cv2
from skimage import io, color

# 1. Cargar la imagen y convertirla a escala de grises (1920 x 1080)
# Reemplaza 'ruta_de_tu_imagen.jpg' por la ubicación real de tu archivo
img_original = io.imread('gengar.png')

if len(img_original.shape) == 3:
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)
else:
    img_gray = img_original

# Asegurar/redimensionar las dimensiones a 1920 x 1080 si fuera necesario
# (alto = 1080, ancho = 1920)
if img_gray.shape != (1080, 1920):
    img_gray = cv2.resize(img_gray, (1920, 1080))

# 2. Calcular la Descomposición en Valores Singulares (SVD)
# U tendrá forma (1080, 1080), S un vector de 1080 valores, Vt forma (1080, 1920)
U, S, Vt = np.linalg.svd(img_gray, full_matrices=False)

# 3. Calcular la suma total D y las proporciones pi
D = np.sum(S)
p_i = S / D

# 4. Calcular la suma acumulada PN para encontrar N (cobertura del 95% = 0.95)
P_N = np.cumsum(p_i)

# Se busca el primer índice donde la suma acumulada alcance o supere 0.95 (+1 por índice 0)
N = np.argmax(P_N >= 0.95) + 1

print(f"Suma total de valores singulares (D): {D:.2f}")
print(f"Número de términos N necesarios para conservar el 95% de la información: {N}")

# 5. Recortar las matrices y los valores singulares a N términos
# U recortada a dimensiones (1080 x N) o (1920 x N) según orientación de ejes
U_N = U[:, :N]              # Mantiene las primeras N columnas
S_N = S[:N]                 # Mantiene los primeros N valores singulares
Vt_N = Vt[:N, :]            # Mantiene las primeras N filas

# Si la imagen se cargó como (1920, 1080) verticalmente:
# U_N tendría dimensiones 1920 x N y Vt_N tendría dimensiones N x 1080

print(f"Forma de U recortada: {U_N.shape}")
print(f"Forma de S recortada: {S_N.shape}")
print(f"Forma de Vt recortada: {Vt_N.shape}")
