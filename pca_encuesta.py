import numpy as np
import pandas as  pd
import plotly.graph_objects as go

# ==========================================
# 1. Cargar datos
# ==========================================
df_original = pd.read_csv('Ejemplo_PCA.csv')

# Quitar la marca de tiempo y enumerar estudiantes cada 
df_original = df_original.drop(columns=['Horodateur']).reset_index(drop=True)
df_original.insert(0, 'Estudiante', [f'Estudiante {i + 1}' for i in range(len(df_original))])

columnas_numericas = ['Edad', 'Estatura', 'Nota Electro', 'Hermanos', 'Peso']

# Se crean 3 versiones, una con todos los datos (crudo), una curando notas mayores a 5.0 como 5.o, y una depurando a los que no saben contestar
df_crudo = df_original.copy()

df_capado = df_original.copy()
df_capado['Nota Electro'] = df_capado['Nota Electro'].clip(upper=5.0)

df_excluido = df_original[df_original['Nota Electro'] <= 5.0].reset_index(drop=True)

versiones = {
    'crudo':    ('Datos originales', df_crudo, '#4C78A8'),
    'capado':   ('Nota > 5.0 capada a 5.0', df_capado, '#54A24B'),
    'excluido': ('Estudiantes con nota > 5.0 excluidos', df_excluido, '#E45756'),
}


# ==========================================
# 2. Estandarizacion de datos y PCA3 
# ==========================================
def pca_3d(df, columnas):
    X = df[columnas].values.astype(float)

    # Estandarizamos porque las variables no están en la misma escala.
    X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    cov = np.cov(X_std, rowvar=False)
    eigval, eigvec = np.linalg.eigh(cov) 

    orden = np.argsort(eigval)[::-1]
    eigval = eigval[orden]
    eigvec = eigvec[:, orden]

    var_exp = eigval / eigval.sum() * 100
    scores = X_std @ eigvec[:, :3]

    return scores, var_exp

#=========================================
# 3. Armamos la grafica
#========================================
archivos=[]

for clave, (titulo, df_v, color) in versiones.items():
    scores, var_exp = pca_3d(df_v, columnas_numericas)
    numeros = [e.split()[-1] for e in df_v['Estudiante']]

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=scores[:, 0],
        y=scores[:, 1],
        z=scores[:, 2],
        mode='markers+text',
        text=numeros,
        textposition='top center',
        marker=dict(
            size=6,
            color=list(range(len(df_v))),
            opacity=0.85
        ),
        hovertemplate=(
            'Estudiante %{text}<br>'
            'PC1: %{x:.3f}<br>'
            'PC2: %{y:.3f}<br>'
            'PC3: %{z:.3f}'
            '<extra></extra>'
        )
    ))

    fig.update_layout(
        title=f'{titulo} — N={len(df_v)}',
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    fig.write_html(
        f'PCA_3d_{clave}.html',
        include_plotlyjs=True
    )
    nombre_archivo = f"PCA_3d_{clave}.html"

    fig.write_html(
        nombre_archivo,
        include_plotlyjs=True
    )
    archivos.append(nombre_archivo)	


print("\n" + "=" * 45)
print("✓ Archivos generados correctamente:")
for archivo in archivos:
    print(f"  • {archivo}")
print("=" * 45)
