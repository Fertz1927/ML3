import numpy as np
import matplotlib.pyplot as plt

sol = np.loadtxt("sol_objects.csv",dtype=str, delimiter=',')
solo = sol[1:,1:]
solo = solo.astype(np.float32)

# ==========================================
# 3. Análisis de Componentes Principales (PCA)
# ==========================================
# Paso A: Centrar los datos (restar la media)
media_solo = np.mean(solo, axis=0)
datos_centrados = solo - media_solo

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


'''
plt.scatter(solo[:, 0], solo[:, 1], alpha=0.5, s=10, color='green', label='Datos transformados')

# Dibujar los eigenvectores (Componentes Principales) escalados por su eigenvalor
# Escalamos para que las flechas sean visibles en el gráfico
origen = media_solo
for i in range(2):
    # Vector escalado (usamos la raíz del eigenvalor para escalar visualmente)
    vector_escalado = eigenvectores_ordenados[:, i] * np.sqrt(eigenvalores_ordenados[i]) * 2.5
    plt.quiver(origen[0], origen[1], vector_escalado[0], vector_escalado[1],
                   color=['red', 'orange'][i], scale=1, scale_units='xy', angles='xy',
                   label=f'PC{i+1} (Varianza: {eigenvalores_ordenados[i]:.2f})')

plt.title('ssss')
plt.xlabel('Eje X\'')
plt.ylabel('Eje Y\'')
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()
'''
