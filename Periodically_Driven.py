import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- 1. PHYSICAL PARAMETERS (Winning Limit Cycle) ---
a_N = 20.0       
b_N = 21.0       
T_N = 0.5        
B_x = -1.0       
B_z = 0.1        # Your winning longitudinal field

B_ext = np.array([B_x, 0.0, B_z])
h = B_ext / np.linalg.norm(B_ext)
alpha_tensor = a_N * np.eye(3) + b_N * np.outer(h, h)

k_scale = 0.7    # Your winning scale

# --- NEW: PERIODIC MODULATION PARAMETERS ---
S0_base = 0.25   # The baseline continuous pump
S_m = 0.1        # The amplitude of the modulation
f_m = 0.35       # The driving frequency in Hz (Estimated near natural frequency)

def get_S0(t):
    """Calculates the periodically modulated pump polarization at time t."""
    return np.array([0.0, 0.0, S0_base + S_m * np.sin(2 * np.pi * f_m * t)])

# --- 2. STATIONARY BLOCH EQUATION (Now Time-Dependent) ---
def calc_electron_spin(t, B_N_flat):
    """Algebraically solves for the electron spin S given the Overhauser field and time t."""
    K = k_scale * (B_ext + B_N_flat)
    
    M_K = np.array([
        [ 0.0,  -K[2],  K[1]],
        [ K[2],  0.0,  -K[0]],
        [-K[1],  K[0],  0.0 ]
    ])
    
    I = np.eye(3)
    # The solver now fetches the specific pump polarization for the current time
    S = np.linalg.solve(I - M_K, get_S0(t))
    return S

# --- 3. DYNAMICAL EQUATION ---
def overhauser_derivative(t, B_N_flat):
    # Pass 't' into the spin calculation
    S = calc_electron_spin(t, B_N_flat)
    dBN_dt = -(1.0 / T_N) * (B_N_flat - alpha_tensor @ S)
    return dBN_dt

# --- 4. SIMULATION EXECUTION ---
t_span = (0, 100)  # Extended time slightly to let the driven system settle
t_eval = np.linspace(0, 100, 5000)

initial_B_N = np.array([0.3, 0.3, 0.3]) # Your winning kick

print("Simulating the Periodically Driven System...")
solution = solve_ivp(
    overhauser_derivative, 
    t_span, 
    initial_B_N, 
    method='RK45', 
    t_eval=t_eval,
    rtol=1e-6, atol=1e-8
)

time = solution.t
B_N_history = solution.y

S_z_history = np.zeros(len(time))
for i in range(len(time)):
    # Calculate Sz using the correct time step
    S = calc_electron_spin(time[i], B_N_history[:, i])
    S_z_history[i] = S[2]

# --- 5. POST-PROCESAMIENTO Y GRÁFICOS ---
transient_cutoff = 40  # Corte transitorio incrementado para limpiar el caos inicial
mask = time > transient_cutoff
t_steady = time[mask]
Sz_steady = S_z_history[mask]

# ── GRÁFICO 1: Dominio del Tiempo (Sistema Forzado) ──
plt.figure(figsize=(8, 5))
plt.plot(t_steady, Sz_steady, color='#d62728', lw=1.5) # Color rojo para el sistema forzado
#plt.title(f"Auto-oscilaciones Forzadas ($f_m$ = {f_m} Hz)", fontsize=14)
plt.xlabel("Tiempo (s)", fontsize=12)
plt.ylabel(r"Polarización del Espín Electrónico $S_z$", fontsize=12)
# plt.xlim(60, 80) # Descomenta esta línea si quieres hacer un zoom a la onda
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('auto_oscilaciones_forzadas.pdf', format='pdf')
plt.show()

# ── GRÁFICO 2: Retrato de Fase (Sistema Forzado) ───────────────────────
plt.figure(figsize=(6, 6)) 
tau = 35 
plt.plot(Sz_steady[:-tau], Sz_steady[tau:], color='black', lw=0.5, alpha=0.7)
#plt.title("Retrato de Fase (Sistema Forzado)", fontsize=14)
plt.xlabel(r"$S_z(t)$", fontsize=12)
plt.ylabel(r"$S_z(t - \tau)$", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('ciclo_limite_forzado.pdf', format='pdf')
plt.show()
