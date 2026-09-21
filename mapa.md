# Mapa Conceptual: Flujos de Trabajo de PCA y SVD en Python

```mermaid
graph TD
    %% Nodo Principal
    A[Procesamiento y Analisis de Datos e Imagenes] --> B[1. PCA Sintetico - cuatro.py]
    A --> C[2. PCA sobre Datos Reales - cinco.py]
    A --> D[3. Decaimiento Espectral SVD]

    %% Rama 1: PCA Sintético
    B --> B1[Entrada: Distribucion Gaussiana 2D]
    B1 --> B2[Transformacion Lineal: Matriz T]
    B2 --> B3[PCA Completo:<br/>- Centrado de datos<br/>- Matriz de covarianza<br/>- Descomposicion np.linalg.eig]
    B3 --> B4[Salidas:<br/>- pca_datos_simulados.png<br/>- resultados_pca_simulado.txt]

    %% Rama 2: PCA Real
    C --> C1[Entrada: sol_objects.csv]
    C1 --> C2[Limpieza Robusta:<br/>- Busqueda automatica de archivo<br/>- RegEx para extraccion de numeros<br/>- Normalizacion de comas a puntos]
    C2 --> C3[PCA y Metricas:<br/>- Eigenvalores y Eigenvectores<br/>- Varianza explicada PC1 y PC2]
    C3 --> C4[Salidas:<br/>- pca_sol_objects.png<br/>- resultados_pca.txt]

    %% Rama 3: SVD y Espectro
    D --> D1[Entrada: Imagen en grises e.g. vicent.png]
    D1 --> D2[Descomposicion SVD:<br/>- Extraccion de Valores Singulares S<br/>- Omision de k=10 iniciales]
    D2 --> D3[Ajuste de Curvas:<br/>- Lineal, Exponencial y Algebraico<br/>- Evaluacion de R2]
    D3 --> D4[Salidas:<br/>- decaimiento_espectro_vicent.png<br/>- resultados_vicent_espectro.txt]

    %% Configuración de Entorno
    A -.-> E[Entorno Headless:<br/>- GitHub Codespaces<br/>- Matplotlib Backend Agg<br/>- Guardado directo a disco]e Entorno
    A -.-> E[<b>Entorno de Ejecución Headless:</b><br/>- GitHub Codespaces<br/>- Backend de Matplotlib: Agg<br/>- Guardado en disco sin interfaz gráfica]
