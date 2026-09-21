import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Modo headless para Codespaces
import matplotlib.pyplot as plt

# ==========================================
# 1. Carga Inteligente con Expresiones Regulares
# ==========================================
nombre_archivo = "sol_objects.csv"

if not os.path.exists(nombre_archivo):
    raise FileNotFoundError(f"No se encontró el archivo '{nombre_archivo}' en el directorio actual.")

filas_procesadas = []

# Expresión regular para capturar cualquier número flotante o entero (incluye negativos y decimales con punto/coma)
patron_numero = re.compile(r'[-+]?\d+(?:[\.,]\d+)?')

with open(nombre_archivo, 'r', encoding='latin-1', errors='ignore') as f:
    lineas = f.readlines()

print(f"--- MUESTRA DEL ARCHIVO (Primeras 3 líneas) ---")
for l in lineas[:3]:
    print(repr(l.strip()))
print("-----------------------------------------------\n")

for linea in lineas:
    # Buscar todos los bloques que parezcan números en la línea
    numeros_encontrados = patron_numero.findall(linea)
    
    # Convertir las cadenas encontradas a float (normalizando coma a punto)
    floats_linea = []
    for num_str in numeros_encontrados:
        try:
            val = float(num_str.replace(',', '.'))
            floats_linea.append(val)
        except ValueError:
            continue
    
    # Si la línea tiene al menos 2 números, tomamos los 2 últimos o primeros según la estructura
    # Omitimos el primer número si parece ser un ID entero secuencial (ej. ID 1, 2, 3...)
    if len(floats_linea) >= 2:
        # Si hay más de 2 números (ej. ID, X, Y), nos quedamos con los 2 últimos que corresponden a las variables
        filas_procesadas.append(floats_linea[-2:])

solo = np.array(filas_procesadas, dtype=np.float32)

if solo.shape[0] == 0:
    raise ValueError("Incapaz de extraer números del archivo. Revisa si el archivo está completamente vacío.")

print(f"¡Éxito total! Matriz procesada correctamente. Dimensiones finales: {solo.shape}")

# ==========================================
# 2. Análisis de Componentes Principales (PCA)
# ==========================================
media_solo = np.mean(solo, axis=0)
datos_centrados = solo - media_solo

matriz_covarianza = np.cov(datos_centrados, rowvar=False)

eigenvalores, eigenvectores = np.linalg.eig(matriz_covarianza)

indices_ordenados = np.argsort(eigenvalores)[::-1]
eigenvalores_ordenados = eigenvalores[indices_ordenados]
eigenvectores_ordenados = eigenvectores[:, indices_ordenados]

varianza_total = np.sum(eigenvalores_ordenados)
pc1_var = (eigenvalores_ordenados[0] / varianza_total) * 100
pc2_var = (eigenvalores_ordenados[1] / varianza_total) * 100

print("\n=== Resultados del PCA ===")
print(f"Eigenvalores: {eigenvalores_ordenados}")
print(f"Varianza explicada por PC1: {pc1_var:.2f}%")
print(f"Varianza explicada por PC2: {pc2_var:.2f}%")

# ==========================================
# 3. Visualización y Guardado (Headless)
# ==========================================
plt.figure(figsize=(8, 6))

plt.scatter(solo[:, 0], solo[:, 1], alpha=0.6, s=15, color='green', label='Datos (sol_objects)')

origen = media_solo
for i in range(2):
    vector_escalado = eigenvectores_ordenados[:, i] * np.sqrt(eigenvalores_ordenados[i]) * 2.5
    plt.quiver(origen[0], origen[1], vector_escalado[0], vector_escalado[1],
               color=['red', 'orange'][i], scale=1, scale_units='xy', angles='xy',
               label=f'PC{i+1} (Varianza: {eigenvalores_ordenados[i]:.2f})')

plt.title('Análisis de Componentes Principales (sol_objects.csv)')
plt.xlabel('Variable X (Columna 1)')
plt.ylabel('Variable Y (Columna 2)')
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()

plt.savefig('pca_sol_objects.png', bbox_inches='tight')
plt.close()

print("\n¡Gráfico del PCA guardado exitosamente como 'pca_sol_objects.png'!")