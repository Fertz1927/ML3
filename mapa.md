# Mapa Conceptual: Flujos de Trabajo de PCA y SVD en Python

```mermaid
graph TD
    %% Nodo Principal
    A[<b>Procesamiento y Análisis de Datos / Imágenes</b>] --> B[<b>1. PCA Sintético (cuatro.py)</b>]
    A --> C[<b>2. PCA sobre Datos Reales (cinco.py)</b>]
    A --> D[<b>3. Decaimiento Espectral SVD</b>]

    %% Rama 1: PCA Sintético
    B --> B1[<b>Entrada:</b> Distribución Gaussiana 2D]
    B1 --> B2[<b>Transformación Lineal:</b> Matriz T]
    B2 --> B3[<b>PCA de 0 a 100:</b><br/>- Centrado de datos<br/>- Matriz de covarianza<br/>- Descomposición espectral np.linalg.eig]
    B3 --> B4[<b>Salidas:</b><br/>- pca_datos_simulados.png<br/>- resultados_pca_simulado.txt]

    %% Rama 2: PCA Real
    C --> C1[<b>Entrada:</b> sol_objects.csv]
    C1 --> C2[<b>Limpieza Robusta:</b><br/>- Búsqueda automática de archivo<br/>- RegEx para extracción de números<br/>- Tolerancia a comillas y decimales de coma]
    C2 --> C3[<b>PCA y Métricas:</b><br/>- Eigenvalores y Eigenvectores<br/>- Varianza explicada por PC1 y PC2]
    C3 --> C4[<b>Salidas:</b><br/>- pca_sol_objects.png<br/>- resultados_pca.txt]

    %% Rama 3: SVD y Espectro
    D --> D1[<b>Entrada:</b> Imagen en escala de grises e.g. vicent.png]
    D1 --> D2[<b>Descomposición SVD:</b><br/>- Extracción de Valores Singulares S<br/>- Omisión de k=10 iniciales]
    D2 --> D3[<b>Ajuste de Curvas (Curve Fitting):</b><br/>- Lineal, Exponencial y Algebraico<br/>- Evaluación del R²]
    D3 --> D4[<b>Salidas:</b><br/>- decaimiento_espectro_vicent.png<br/>- resultados_vicent_espectro.txt]

    %% Configuración de Entorno
    A -.-> E[<b>Entorno de Ejecución Headless:</b><br/>- GitHub Codespaces<br/>- Backend de Matplotlib: Agg<br/>- Guardado en disco sin interfaz gráfica]