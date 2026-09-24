import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Configuración para entorno headless (Codespaces)
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuración y Generación de Datos
# ==========================================
N = 1000          # Número de muestras
sigma_x = 2.0     # Desviación estándar en X
sigma_y = 0.5     # Desviación estándar en Y

media = [0, 0]
covarianza_inicial = [[sigma_x**2, 0], [0, sigma_y**2]]

# Semilla fija para reproducibilidad exacta
np.random.seed(42)
datos_originales = np.random.multivariate_normal(media, covarianza_inicial, N)

# ==========================================
# 2. Aplicación de la Matriz T
# ==========================================
T = np.array([[1, 2],
              [-1, 0]])

datos_transformados = datos_originales @ T.T

# ==========================================
# 3. Análisis de Componentes Principales (PCA)
# ==========================================
media_transformada = np.mean(datos_transformados, axis=0)
datos_centrados = datos_transformados - media_transformada

matriz_covarianza = np.cov(datos_centrados, rowvar=False)

eigenvalores, eigenvectores = np.linalg.eigh(matriz_covarianza)

indices_ordenados = np.argsort(eigenvalores)[::-1]
eigenvalores_ordenados = eigenvalores[indices_ordenados]
eigenvectores_ordenados = eigenvectores[:, indices_ordenados]

# DEFINICIÓN DE VARIANZA TOTAL (Corrigiendo el NameError)
varianza_total = np.sum(eigenvalores_ordenados)
pc1_var = (eigenvalores_ordenados[0] / varianza_total) * 100
pc2_var = (eigenvalores_ordenados[1] / varianza_total) * 100

print("=== Resultados del PCA ===")
print(f"Matriz de Covarianza:\n{matriz_covarianza}\n")
print(f"Eigenvalores (Varianza explicada): {eigenvalores_ordenados}")
print(f"Eigenvectores (Componentes Principales):\n{eigenvectores_ordenados}")
print(f"Varianza total explicada: {varianza_total:.4f}")
print(f"Varianza explicada por PC1: {pc1_var:.2f}%")
print(f"Varianza explicada por PC2: {pc2_var:.2f}%")

# ==========================================
# 4. Visualización y Guardado de Imagen
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico 1: Datos Originales
axes[0].scatter(datos_originales[:, 0], datos_originales[:, 1], alpha=0.5, s=10, color='blue')
axes[0].set_title('Distribución Gaussiana Original')
axes[0].set_xlabel('Eje X')
axes[0].set_ylabel('Eje Y')
axes[0].axis('equal')
axes[0].grid(True, linestyle='--', alpha=0.6)

# Gráfico 2: Datos Transformados + PCA
axes[1].scatter(datos_transformados[:, 0], datos_transformados[:, 1], alpha=0.5, s=10, color='green', label='Datos transformados')

origen = media_transformada
for i in range(2):
    vector_escalado = eigenvectores_ordenados[:, i] * np.sqrt(eigenvalores_ordenados[i]) * 2.5
    axes[1].quiver(origen[0], origen[1], vector_escalado[0], vector_escalado[1],
                   color=['red', 'orange'][i], scale=1, scale_units='xy', angles='xy',
                   label=f'PC{i+1} (Varianza: {eigenvalores_ordenados[i]:.2f})')

axes[1].set_title('Datos Transformados por T y PCA')
axes[1].set_xlabel("Eje X'")
axes[1].set_ylabel("Eje Y'")
axes[1].axis('equal')
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend()

plt.tight_layout()

# Guardar figura en disco
plt.savefig('pca_datos_simulados.png', bbox_inches='tight')
plt.close()

# ==========================================
# 5. Exportación de Resultados a Archivo de Texto (.txt)
# ==========================================
archivo_salida_txt = "resultados_pca_simulado.txt"

with open(archivo_salida_txt, "w", encoding="utf-8") as archivo:
    archivo.write("==================================================\n")
    archivo.write("     RESULTADOS DEL PCA (DATOS GAUSSIANOS + T)     \n")
    archivo.write("==================================================\n\n")
    
    archivo.write("--- PARÁMETROS DE SIMULACIÓN ---\n")
    archivo.write(f"Número de muestras (N): {N}\n")
    archivo.write(f"Sigma X inicial: {sigma_x}\n")
    archivo.write(f"Sigma Y inicial: {sigma_y}\n")
    archivo.write(f"Matriz T aplicada:\n{T}\n\n")
    
    archivo.write("--- MATRIZ DE COVARIANZA (DATOS TRANSFORMADOS) ---\n")
    archivo.write(f"{matriz_covarianza}\n\n")
    
    archivo.write("--- EIGENVALORES Y EIGENVECTORES ---\n")
    archivo.write(f"Eigenvalores (Varianzas): {eigenvalores_ordenados}\n\n")
    archivo.write("Eigenvectores (Direcciones de los Componentes):\n")
    archivo.write(f"PC1 Vector: {eigenvectores_ordenados[:, 0]}\n")
    archivo.write(f"PC2 Vector: {eigenvectores_ordenados[:, 1]}\n\n")
    
    archivo.write("--- VARIANZA EXPLICADA ---\n")
    archivo.write(f"Varianza total explicada: {varianza_total:.4f}\n")
    archivo.write(f"Varianza explicada por PC1: {pc1_var:.2f}%\n")
    archivo.write(f"Varianza explicada por PC2: {pc2_var:.2f}%\n")

print("\n¡Ejecución completada con éxito!")
print("1. Imagen guardada como: 'pca_datos_simulados.png'")
print(f"2. Reporte guardado como: '{archivo_salida_txt}'")
