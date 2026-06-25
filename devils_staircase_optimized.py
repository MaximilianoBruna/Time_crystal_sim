import numpy as np
import matplotlib.pyplot as plt

# ── 1. PARÁMETROS FÍSICOS (sin cambios) ──────────────────────────────────────
a_N, b_N, T_N, B_x, B_z = 20.0, 21.0, 0.5, -1.0, 0.1
B_ext  = np.array([B_x, 0.0, B_z])
h_hat  = B_ext / np.linalg.norm(B_ext)
alpha  = a_N * np.eye(3) + b_N * np.outer(h_hat, h_hat)
k_scale, S0_base, S_m = 0.7, 0.25, 0.08

# ── 2. DERIVADA EN LOTE ───────────────────────────────────────────────────────
# CAMBIO CLAVE: en lugar de resolver UN sistema a la vez dentro de 70 llamadas
# separadas a solve_ivp, esta función resuelve TODOS los fm simultáneamente.
#
# BN     : (N_fm, 3)  — estado de todos los fm en este instante
# fm_arr : (N_fm,)    — todas las frecuencias de modulación
# Returns: (N_fm, 3)  — dBN/dt para cada frecuencia a la vez

def F_batch(t, BN, fm_arr):
    N = len(fm_arr)
    K = k_scale * (B_ext + BN)         # (N_fm, 3) — B_ext hace broadcast

    # (I − M_K) para todos los N_fm de una vez: shape (N_fm, 3, 3)
    A = np.empty((N, 3, 3))
    A[:, 0, 0] =  1.0;      A[:, 0, 1] =  K[:, 2]; A[:, 0, 2] = -K[:, 1]
    A[:, 1, 0] = -K[:, 2]; A[:, 1, 1] =  1.0;      A[:, 1, 2] =  K[:, 0]
    A[:, 2, 0] =  K[:, 1]; A[:, 2, 1] = -K[:, 0];  A[:, 2, 2] =  1.0

    # S0 depende de (fm, t): vectorizado sobre todos los fm a la vez
    S0 = np.zeros((N, 3))
    S0[:, 2] = S0_base + S_m * np.sin(2 * np.pi * fm_arr * t)

    S  = np.linalg.solve(A, S0)           # (N_fm, 3)
    aS = (alpha @ S.T).T                   # (N_fm, 3) — alpha⋅S para cada fm
    return -(1.0 / T_N) * (BN - aS)

# ── 3. INTEGRADOR RK4 EN LOTE (paso fijo) ────────────────────────────────────
# Un solo bucle Python integra TODOS los fm en paralelo.
# Se guarda sólo la ventana de estado estacionario (t ≥ t_steady).
#
# Complejidad total:
#   Original : 70 × n_steps × 15 evaluaciones (Radau) × costo_1fm
#   Nuevo    :  1 × n_steps ×  4 evaluaciones (RK4)   × costo_70fm
#
# LAPACK vectoriza las 70 resoluciones 3×3 de forma nativa → 15–30× más rápido.

def rk4_batch(fm_array, t_end=600.0, t_steady=300.0, pts_per_period=40):
    N_fm    = len(fm_array)
    dt      = (1.0 / fm_array.max()) / pts_per_period  # paso más fino requerido
    n_steps = int(np.ceil(t_end / dt))
    n_save  = int(np.ceil((t_end - t_steady) / dt)) + 2

    BN = np.full((N_fm, 3), 0.3)          # condición inicial para todos los fm
    t_buf  = np.empty(n_save)
    BN_buf = np.empty((n_save, N_fm, 3))
    idx, t = 0, 0.0

    for _ in range(n_steps):
        if t >= t_steady:
            t_buf[idx]  = t
            BN_buf[idx] = BN               # copia de valores (no referencia)
            idx += 1
        k1 = F_batch(t,       BN,              fm_array)
        k2 = F_batch(t+dt/2,  BN + dt/2 * k1, fm_array)
        k3 = F_batch(t+dt/2,  BN + dt/2 * k2, fm_array)
        k4 = F_batch(t+dt,    BN + dt   * k3, fm_array)
        BN += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t  += dt

    return t_buf[:idx], BN_buf[:idx]       # (T_save,), (T_save, N_fm, 3)

# ── 4. Sz VECTORIZADO SOBRE TODOS LOS (tiempo, fm) A LA VEZ ─────────────────
# Una sola llamada a np.linalg.solve para T_save × N_fm ≈ 79 000 sistemas 3×3.
# Reemplaza el bucle Python de 5 000 iteraciones del original.

def calc_Sz(t_arr, BN_all, fm_array):
    T, N_fm = BN_all.shape[:2]
    BN_flat = BN_all.reshape(T * N_fm, 3)
    K = k_scale * (B_ext + BN_flat)        # (T*N_fm, 3)

    A = np.empty((T * N_fm, 3, 3))
    A[:, 0, 0] =  1.0;      A[:, 0, 1] =  K[:, 2]; A[:, 0, 2] = -K[:, 1]
    A[:, 1, 0] = -K[:, 2]; A[:, 1, 1] =  1.0;      A[:, 1, 2] =  K[:, 0]
    A[:, 2, 0] =  K[:, 1]; A[:, 2, 1] = -K[:, 0];  A[:, 2, 2] =  1.0

    # S0(t_i, fm_j): broadcasting (T,1) × (1,N_fm) → (T,N_fm) → aplanar
    sin_mat  = np.sin(2*np.pi * t_arr[:, None] * fm_array[None, :])  # (T, N_fm)
    S0 = np.zeros((T * N_fm, 3))
    S0[:, 2] = (S0_base + S_m * sin_mat).ravel()

    return np.linalg.solve(A, S0)[:, 2].reshape(T, N_fm)  # (T, N_fm) Sz values


# ── 5. NÚMEROS DE ENROLLAMIENTO (Vía FFT) ────────────────────────────────────
def winding_numbers(t_arr, Sz_all, fm_array):
    W = np.zeros(len(fm_array))
    
    # Asumimos pasos de tiempo uniformes desde el integrador RK4
    dt = t_arr[1] - t_arr[0]
    n_samples = len(t_arr)
    
    pad_factor = 10 
    n_padded = n_samples * pad_factor
    
    # Eje de frecuencias para el FFT
    freqs = np.fft.rfftfreq(n_padded, d=dt)
    
    for j, fm in enumerate(fm_array):
        Sz = Sz_all[:, j]
        # Centrar la señal en cero elimina el pico de frecuencia 0 (DC)
        Szc = Sz - np.mean(Sz)
        
        # Calcular el FFT real con zero-padding
        fft_vals = np.fft.rfft(Szc, n=n_padded)
        power_spectrum = np.abs(fft_vals)**2
        
        # Encontrar el índice del pico dominante de energía
        peak_idx = np.argmax(power_spectrum)
        f_AO = freqs[peak_idx]
        
        # Calcular número de enrollamiento
        W[j] = f_AO / fm
        print(f"fm={fm:.3f} Hz  →  f_AO={f_AO:.3f} Hz  →  W={W[j]:.4f}")
        
    return W

# ── 6. EJECUCIÓN ─────────────────────────────────────────────────────────────
fm_array = np.linspace(0.14, 0.85, 300)
pts_per_period = 60
t_steady = 300.0
t_end = 800.0
dt_used  = (1.0 / fm_array.max()) / pts_per_period
n_steps  = int(np.ceil(t_end / dt_used))

print(f"RK4 en lote — {len(fm_array)} frecuencias × {n_steps} pasos "
      f"(dt = {dt_used:.4f} s)")

t_arr, BN_all = rk4_batch(fm_array, t_end=t_end, t_steady=t_steady, pts_per_period=pts_per_period)
print(f"Integración lista. {len(t_arr)} puntos de estado estacionario guardados.")

print("Calculando Sz (lote vectorizado)...")
Sz_all = calc_Sz(t_arr, BN_all, fm_array)

print("Calculando números de enrollamiento...")
W = winding_numbers(t_arr, Sz_all, fm_array)

# ── 7. GRÁFICO ────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.plot(fm_array, W, 'o-', color="#d93d05", markersize=6, lw=2.0) 
#plt.title("Escalera del Diablo: Lenguas de Arnold 1/2 y 2/3", fontsize=14)
plt.xlabel(r"Frecuencia de Modulación del Láser $f_m$ (Hz)", fontsize=12)
plt.ylabel(r"Número de Enrollamiento $W$", fontsize=12)
plt.axhline(y=2/3, color='gray', ls='--', alpha=0.5)
plt.axhline(y=0.5, color='gray', ls=':',  alpha=0.5)

plt.yticks([0.0, 0.25, 0.333, 0.4, 0.5, 0.666, 0.75, 1.0], 
           ['0 (Muerto)', '1/4', '1/3', '2/5', '1/2', '2/3', '3/4', '1/1'])
plt.xlim(0.13,0.7)
plt.ylim(0.1, 1.2)
plt.grid(True, ls=':', alpha=0.7)
plt.tight_layout()
plt.savefig('Devil_Staircase.pdf', format='pdf')
plt.show()
