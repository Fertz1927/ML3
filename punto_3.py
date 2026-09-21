import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

img = cv2.imread('vicent.png', cv2.IMREAD_GRAYSCALE)

U, S, Vt = np.linalg.svd(img, full_matrices=False)


k_omitir = 10
indices = np.arange(k_omitir, len(S)) # Índices i = k_omitir, ..., M
S_recortado = S[k_omitir:] # Ignorar los primeros valores singulares.

# A. Ajuste Lineal: S = a * i + b
p_lineal = np.polyfit(indices, S_recortado, 1)
S_fit_lineal = np.polyval(p_lineal, indices)

# B. Ajuste Exponencial: ln(S) = ln(a) - b * i  =>  S = a * e^(-b * i)
p_exp = np.polyfit(indices, np.log(S_recortado), 1)
S_fit_exp = np.exp(p_exp[1]) * np.exp(p_exp[0] * indices)

# C. Ajuste Algebraico (Ley de Potencias): ln(S) = ln(a) - b * ln(i)  =>  S = a * i^(-b)
p_alg = np.polyfit(np.log(indices), np.log(S_recortado), 1)
S_fit_alg = np.exp(p_alg[1]) * (indices ** p_alg[0])

def calcular_r2(y_real, y_pred):
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    return 1 - (ss_res / ss_tot)

r2_lineal = calcular_r2(S_recortado, S_fit_lineal)
r2_exp = calcular_r2(S_recortado, S_fit_exp)
r2_alg = calcular_r2(S_recortado, S_fit_alg)

print("=== EVALUACIÓN DE DECAIMIENTO DEL ESPECTRO SVD ===")
print(f"Valores singulares evaluados: a partir del índice {k_omitir}")
print(f"R² Ajuste Lineal:      {r2_lineal:.4f}")
print(f"R² Ajuste Exponencial: {r2_exp:.4f}")
print(f"R² Ajuste Algebraico:  {r2_alg:.4f}")

resultados = {
    "Lineal": r2_lineal,
    "Exponencial": r2_exp,
    "Algebraico (Ley de Potencias)": r2_alg
}
mejor_modelo = max(resultados, key=resultados.get)
print(f"\nEl decaimiento que mejor describe la cola del espectro es: **{mejor_modelo}** con R² = {resultados[mejor_modelo]:.4f}")

# Grafica.
plt.figure(figsize=(10, 6))
plt.plot(indices, S_recortado, 'k.', label='Valores Singulares (reales)', alpha=0.5)
plt.plot(indices, S_fit_lineal, 'r--', label=f'Lineal (R²={r2_lineal:.2f})')
plt.plot(indices, S_fit_exp, 'g--', label=f'Exponencial (R²={r2_exp:.2f})')
plt.plot(indices, S_fit_alg, 'b--', label=f'Algebraico (R²={r2_alg:.2f})')

plt.yscale('log') # Escala logarítmica en Y para visualizar mejor el decaimiento
plt.title("Espectro de Valores Singulares y Ajuste de Decaimiento (vicent.png)")
plt.xlabel("Índice del Valor Singular (i)")
plt.ylabel("Magnitud Singular (Escala Log)")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.savefig('decaimiento_espectro_vicent.png', bbox_inches='tight')
plt.close()

print("\n¡Gráfica 'decaimiento_espectro_vicent.png' guardada exitosamente!")