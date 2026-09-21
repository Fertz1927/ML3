# Mapa Conceptual: Análisis Dimensional y Ajuste Espectral

```mermaid
graph TD
    A[Analisis de Datos Multivariados e Imagenes] --> B[Analisis de Componentes Principales - PCA]
    A --> C[Descomposicion en Valores Singulares - SVD]

    B --> B1[Sintesis de Datos]
    B1 --> B2[Distribuciones Gaussianas 2D]
    B2 --> B3[Transformaciones Afines y Lineales]

    B --> B4[Procesamiento de Datos Reales]
    B4 --> B5[Limpieza y Normalizacion RegEx]
    B5 --> B6[Estructuras Tabulares Inconsistentes]

    B --> B7[Algebra Lineal y Estadistica]
    B7 --> B8[Centrado de Datos y Media]
    B8 --> B9[Matriz de Covarianza]
    B9 --> B10[Eigenvalores y Eigenvectores]
    B10 --> B11[Varianza Explicada Acumulada]

    C --> C1[Transformacion de Dominio de Imagen]
    C1 --> C2[Matriz de Luminancia en Grises]
    C2 --> C3[Espectro de Valores Singulares]

    C --> C4[Modelado y Decaimiento Espectral]
    C4 --> C5[Filtrado de Bajas Frecuencias]
    C5 --> C6[Ajuste de Modelos: Lineal, Exponencial y Potencias]
    C6 --> C7[Evaluacion de Calidad via Coeficiente R2]

    A -.-> D[Entorno Metodologico]
    D --> D1[Renderizado sin Interfaz - Headless]
    D1 --> D2[Exportacion Sistematica de Graficos y Reportes]
