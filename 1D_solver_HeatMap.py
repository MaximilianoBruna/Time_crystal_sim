import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

#  PARÁMETROS 
a_N, b_N, T_N, B_x, B_z = 20.0, 21.0, 0.5, -1.0, 0.1
B_ext = np.array([B_x, 0.0, B_z])
h = B_ext / np.linalg.norm(B_ext)
alpha_tensor = a_N * np.eye(3) + b_N * np.outer(h, h)
k_scale = 0.7

#  GRILLA ESPACIAL 
N = 41
x = np.linspace(-5, 5, N)
dx = x[1] - x[0]
D = 1.0

S0_max, sigma = 0.33, 1.5
S0_z = S0_max * np.exp(-x**2 / (2 * sigma**2))
S0_array = np.zeros((3, N))
S0_array[2, :] = S0_z

B_ext_col = B_ext[:, np.newaxis]
S0_T      = np.ascontiguousarray(S0_array.T)  # (N, 3) — C-contiguo
inv_TN    = 1.0 / T_N
inv_dx2   = D / dx**2

# ── PATRÓN DE DISPERSIÓN DEL JACOBIANO ────────────────────────────────────
# Índice de variable: c*N + n  (componente c=0..2, punto espacial n=0..N-1)
#
# d(B_N[c,n])/dt depende de:
#   • B_N[c', n] para c'=0,1,2  → acoplamiento local vía solve(I-M_K) y alpha
#   • B_N[c, n±1]               → difusión (misma componente, puntos adyacentes)
#
# Eso son ≤5 entradas no nulas por fila en lugar de 123.
# Radau sólo necesita ~5 evaluaciones de la derivada por columna del Jacobiano
# en vez de 123 → speedup ~25x sólo en el cálculo del Jacobiano.

jac_sp = lil_matrix((3 * N, 3 * N), dtype=np.float64)
for c in range(3):
    for n in range(N):
        row = c * N + n
        for c2 in range(3):           # bloque 3×3 diagonal en espacio
            jac_sp[row, c2 * N + n] = 1
        if n > 0:                     # vecino izquierdo (difusión)
            jac_sp[row, c * N + n - 1] = 1
        if n < N - 1:                 # vecino derecho (difusión)
            jac_sp[row, c * N + n + 1] = 1
jac_sparsity = jac_sp.tocsc()

print(f"Entradas no nulas en el Jacobiano: {jac_sparsity.nnz} de {(3*N)**2} "
      f"({100*jac_sparsity.nnz/(3*N)**2:.1f}% del total)")

# ── DERIVADA PDE VECTORIZADA ──────────────────────────────────────────────
# np.linalg.solve(A, b) con A:(N,3,3) y b:(N,3)

def pde_derivative(t, BN_flat):
    B_N = BN_flat.reshape((3, N))

    # K = k_scale*(B_ext + B_N[:,i]) vectorizado sobre N puntos
    K = k_scale * (B_ext_col + B_N)   # (3, N)

    # (I - M_K) para cada punto: [[1, Kz,-Ky],[-Kz,1,Kx],[Ky,-Kx,1]]
    ones = np.ones(N)
    A = np.empty((N, 3, 3))
    A[:, 0, 0] =  ones;  A[:, 0, 1] =  K[2]; A[:, 0, 2] = -K[1]
    A[:, 1, 0] = -K[2]; A[:, 1, 1] =  ones;  A[:, 1, 2] =  K[0]
    A[:, 2, 0] =  K[1]; A[:, 2, 1] = -K[0];  A[:, 2, 2] =  ones

    # Solución en lote (N,3,3)\(N,3) → (N,3) → transponer a (3,N)
    S = np.linalg.solve(A, S0_T).T    # (3, N)

    # Dinámica nuclear local
    local_dynamics = -inv_TN * (B_N - alpha_tensor @ S)

    # Laplaciano discreto — las 3 componentes a la vez (sin bucle j)
    diffusion = np.empty_like(B_N)
    diffusion[:, 1:-1] = inv_dx2 * (B_N[:, 2:] - 2*B_N[:, 1:-1] + B_N[:, :-2])
    diffusion[:, 0]    = inv_dx2 * (B_N[:, 1]  -   B_N[:, 0])
    diffusion[:, -1]   = inv_dx2 * (B_N[:, -2] -   B_N[:, -1])

    return (local_dynamics + diffusion).ravel()

# ── EJECUCIÓN ─────────────────────────────────────────────────────────────
t_span = (0, 30)
t_eval = np.linspace(0, 30, 10000)
initial_BN = np.ones((3, N)) * 0.01

print(f"Resolviendo PDE 1D con {N} puntos espaciales...")
sol = solve_ivp(
    pde_derivative,
    t_span,
    initial_BN.ravel(),
    method='RK45', 
    t_eval=t_eval,
    rtol=1e-4, atol=1e-6,
    jac_sparsity=jac_sparsity,        
                                       
)
print(f"Completado en {sol.nfev} evaluaciones. Éxito: {sol.success}")

# ── POST-PROCESAMIENTO VECTORIZADO ─────────────────────────────────────────
T_steps = len(sol.t)
B_N_all = sol.y.reshape((3, N, T_steps))                         # (3, N, T)

K_all = k_scale * (B_ext[:, np.newaxis, np.newaxis] + B_N_all)   # (3, N, T)
NT    = N * T_steps
K_f   = K_all.reshape(3, NT)                                      # (3, NT)
# Con reshape en orden C: índice n*T_steps + t → (punto n, instante t)

ones_f = np.ones(NT)
A_all = np.empty((NT, 3, 3))
A_all[:, 0, 0] =  ones_f; A_all[:, 0, 1] =  K_f[2]; A_all[:, 0, 2] = -K_f[1]
A_all[:, 1, 0] = -K_f[2]; A_all[:, 1, 1] =  ones_f; A_all[:, 1, 2] =  K_f[0]
A_all[:, 2, 0] =  K_f[1]; A_all[:, 2, 1] = -K_f[0]; A_all[:, 2, 2] =  ones_f

# np.repeat repite cada fila de S0_T exactamente T_steps veces → índice n*T+t
S0_tiled  = np.repeat(S0_T, T_steps, axis=0)                     # (NT, 3)
Sz_history = np.linalg.solve(A_all, S0_tiled)[:, 2].reshape(N, T_steps)

# ── GRÁFICO ────────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 6))
plt.contourf(sol.t, x, Sz_history, levels=50, cmap='magma')

plt.colorbar(label=r'Polarización del Espín Electrónico $S_z$')

#plt.title("Cristal de Tiempo Espacial 1D: Ondas de Sincronización", fontsize=15)
plt.xlabel("Tiempo (s)", fontsize=12)
plt.ylabel(r"Distancia desde el Centro del Láser $x$ ($\mu$m)", fontsize=12)

plt.axhline(0, color='white', linestyle='--', alpha=0.3)
plt.tight_layout()

plt.savefig('mapa_calor.pdf', format='pdf')
plt.show()
