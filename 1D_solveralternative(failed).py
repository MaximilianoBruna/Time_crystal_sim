import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ── 1. PARÁMETROS FÍSICOS (sin cambios) ──────────────────────────────────────
a_N, b_N, T_N, B_x, B_z = 20.0, 21.0, 0.5, -1.0, 0.1
B_ext = np.array([B_x, 0.0, B_z])
h = B_ext / np.linalg.norm(B_ext)
alpha_tensor = a_N * np.eye(3) + b_N * np.outer(h, h)
k_scale = 0.7

# ── 2. GRILLA ESPACIAL (sin cambios) ─────────────────────────────────────────
N = 41
x = np.linspace(-5, 5, N)
dx = x[1] - x[0]
D = 1.0

S0_max, sigma = 0.26, 1.5
S0_z = S0_max * np.exp(-x**2 / (2 * sigma**2))
S0_array = np.zeros((3, N))
S0_array[2, :] = S0_z

# ── PRECÓMPUTOS (constantes fuera del bucle del solver) ──────────────────────
# Evita recalcular estas formas en cada llamada a pde_derivative.
B_ext_col = B_ext[:, np.newaxis]         # (3,1): broadcast sobre N puntos
S0_T = np.ascontiguousarray(S0_array.T)  # (N,3): C-contiguo para linalg.solve
inv_TN = 1.0 / T_N
inv_dx2 = D / dx**2

# ── 3. DERIVADA PDE OPTIMIZADA ───────────────────────────────────────────────
# CAMBIO CLAVE: se elimina el bucle Python "for i in range(N)".
# np.linalg.solve(A, b) con A:(N,3,3) y b:(N,3) resuelve los N sistemas
# a la vez usando LAPACK, que es ~20-40x más rápido que el bucle original.

def pde_derivative(t, BN_flat):
    B_N = BN_flat.reshape((3, N))

    # K_i = k_scale*(B_ext + B_N[:,i]) — vectorizado sobre los N puntos
    K = k_scale * (B_ext_col + B_N)   # (3, N)

    # (I - M_K) para cada punto: [[1, Kz, -Ky], [-Kz, 1, Kx], [Ky, -Kx, 1]]
    # Construcción directa sin bucle ni np.array([...]) que crearía copias
    ones = np.ones(N)
    A = np.empty((N, 3, 3))
    A[:, 0, 0] =  ones;   A[:, 0, 1] =  K[2]; A[:, 0, 2] = -K[1]
    A[:, 1, 0] = -K[2];  A[:, 1, 1] =  ones;  A[:, 1, 2] =  K[0]
    A[:, 2, 0] =  K[1];  A[:, 2, 1] = -K[0];  A[:, 2, 2] =  ones

    # Solución en lote: (N,3,3) \ (N,3) → (N,3) → transponer a (3,N)
    S = np.linalg.solve(A, S0_T).T   # (3, N)

    # Dinámica nuclear local
    local_dynamics = -inv_TN * (B_N - alpha_tensor @ S)

    # Laplaciano discreto — las 3 componentes a la vez (sin bucle j)
    diffusion = np.empty_like(B_N)
    diffusion[:, 1:-1] = inv_dx2 * (B_N[:, 2:] - 2*B_N[:, 1:-1] + B_N[:, :-2])
    diffusion[:, 0]    = inv_dx2 * (B_N[:, 1]  -   B_N[:, 0])
    diffusion[:, -1]   = inv_dx2 * (B_N[:, -2] -   B_N[:, -1])

    return (local_dynamics + diffusion).ravel()


# ── 4. EJECUCIÓN ─────────────────────────────────────────────────────────────
t_span = (0, 30)
t_eval = np.linspace(0, 30, 600)
initial_BN = np.ones((3, N)) * 0.01

print(f"Resolviendo PDE 1D con {N} puntos espaciales...")
sol = solve_ivp(
    pde_derivative,
    t_span,
    initial_BN.ravel(),
    method='Radau',
    t_eval=t_eval,
    rtol=1e-4, atol=1e-6,
)
print(f"Completado en {sol.nfev} evaluaciones. Éxito: {sol.success}")

# ── 5. POST-PROCESAMIENTO VECTORIZADO ─────────────────────────────────────────
# CAMBIO CLAVE: se elimina el doble bucle anidado (600×41 llamadas a solve).
# En su lugar, se construye un único lote de NT=24600 sistemas y se resuelven
# todos en una sola llamada a np.linalg.solve.

T_steps = len(sol.t)
B_N_all = sol.y.reshape((3, N, T_steps))                         # (3, N, T)

# K para todos los (punto, tiempo) a la vez
K_all = k_scale * (B_ext[:, np.newaxis, np.newaxis] + B_N_all)   # (3, N, T)
NT = N * T_steps
K_f = K_all.reshape(3, NT)                                        # (3, NT)
# Nota: con reshape en orden C, el índice n*T_steps + t
#       corresponde a (punto espacial n, instante t)

# Lote completo de matrices (I - M_K)
ones_f = np.ones(NT)
A_all = np.empty((NT, 3, 3))
A_all[:, 0, 0] =  ones_f;  A_all[:, 0, 1] =  K_f[2]; A_all[:, 0, 2] = -K_f[1]
A_all[:, 1, 0] = -K_f[2]; A_all[:, 1, 1] =  ones_f;  A_all[:, 1, 2] =  K_f[0]
A_all[:, 2, 0] =  K_f[1];  A_all[:, 2, 1] = -K_f[0]; A_all[:, 2, 2] =  ones_f

# S0 repetido para cada instante (np.repeat repite cada fila T_steps veces
# consecutivamente, manteniéndola alineada con el índice n*T_steps + t)
S0_tiled = np.repeat(S0_T, T_steps, axis=0)                      # (NT, 3)

# Una sola llamada batch → extraer componente Sz → reshapear a (N, T)
Sz_history = np.linalg.solve(A_all, S0_tiled)[:, 2].reshape(N, T_steps)

# ── 6. GRÁFICO ────────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 6))
plt.contourf(sol.t, x, Sz_history, levels=50, cmap='magma')
plt.colorbar(label='Electron Spin Polarization $S_z$')
plt.title("1D Spatial Time Crystal: Synchronization Waves", fontsize=15)
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel(r"Distance from Laser Center $x$ ($\mu$m)", fontsize=12)
plt.axhline(0, color='white', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# ── OPCIONAL: Si necesitas aún más velocidad ──────────────────────────────────
# Instala Numba (pip install numba) y envuelve el bucle de la derivada con
# @njit(parallel=True) + prange(N) para paralelizar sobre los núcleos de CPU.
# Alternativamente, prueba method='BDF' en lugar de 'Radau' — a veces es
# más rápido para este nivel de rigidez (stiffness ratio ≈ D/dx² / (1/T_N) ≈ 8).