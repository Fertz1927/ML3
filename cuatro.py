import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuración y Generación de Datos
# ==========================================
# Definimos los parámetros (puedes cambiarlos)
N = 1000          # Número de muestras
sigma_x = 2.0     # Desviación estándar en X
sigma_y = 0.5     # Desviación estándar en Y

# Generador de números aleatorios con distribución gaussiana 2D centrada en cero
# La matriz de covarianza es diagonal: [[sigma_x^2, 0], [0, sigma_y^2]]
media = [0, 0]
covarianza_inicial = [[sigma_x**2, 0], [0, sigma_y**2]]

# Generamos las parejas ordenadas (N, 2)
datos_originales = np.random.multivariate_normal(media, covarianza_inicial, N)

# ==========================================
# 2. Aplicación de la Matriz T
# ==========================================
# Matriz de transformación T
T = np.array([[1, 2],
              [-1, 0]])

# Para aplicar T a cada pareja ordenada (fila), multiplicamos: X_trans = X @ T.T
# Si X es un vector fila [x, y], entonces [x, y] * T.T = [x + 2y, -x]
datos_transformados = datos_originales @ T.T

# ==========================================
# 3. Análisis de Componentes Principales (PCA)
# ==========================================
# Paso A: Centrar los datos (restar la media)
media_transformada = np.mean(datos_transformados, axis=0)
datos_centrados = datos_transformados - media_transformada

# Paso B: Calcular la matriz de covarianza de los datos transformados
# rowvar=False indica que las columnas son las variables (X e Y)
matriz_covarianza = np.cov(datos_centrados, rowvar=False)

# Paso C: Calcular eigenvalores y eigenvectores
# Los eigenvectores son las direcciones de los componentes principales
# Los eigenvalores indican la varianza explicada por cada componente
eigenvalores, eigenvectores = np.linalg.eig(matriz_covarianza)

# Paso D: Ordenar los eigenvalores y eigenvectores de mayor a menor
indices_ordenados = np.argsort(eigenvalores)[::-1]
eigenvalores_ordenados = eigenvalores[indices_ordenados]
eigenvectores_ordenados = eigenvectores[:, indices_ordenados]

print("=== Resultados del PCA ===")
print(f"Matriz de Covarianza:\n{matriz_covarianza}\n")
print(f"Eigenvalores (Varianza explicada): {eigenvalores_ordenados}")
print(f"Eigenvectores (Componentes Principales):\n{eigenvectores_ordenados}")
print(f"Varianza total explicada: {np.sum(eigenvalores_ordenados):.4f}")
print(f"Varianza explicada por PC1: {(eigenvalores_ordenados[0]/np.sum(eigenvalores_ordenados))*100:.2f}%")
print(f"Varianza explicada por PC2: {(eigenvalores_ordenados[1]/np.sum(eigenvalores_ordenados))*100:.2f}%")

# ==========================================
# 4. Visualización
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

# Dibujar los eigenvectores (Componentes Principales) escalados por su eigenvalor
# Escalamos para que las flechas sean visibles en el gráfico
origen = media_transformada
for i in range(2):
    # Vector escalado (usamos la raíz del eigenvalor para escalar visualmente)
    vector_escalado = eigenvectores_ordenados[:, i] * np.sqrt(eigenvalores_ordenados[i]) * 2.5
    axes[1].quiver(origen[0], origen[1], vector_escalado[0], vector_escalado[1],
                   color=['red', 'orange'][i], scale=1, scale_units='xy', angles='xy',
                   label=f'PC{i+1} (Varianza: {eigenvalores_ordenados[i]:.2f})')

axes[1].set_title('Datos Transformados por T y PCA')
axes[1].set_xlabel('Eje X\'')
axes[1].set_ylabel('Eje Y\'')
axes[1].axis('equal')
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend()

plt.tight_layout()
plt.show()
