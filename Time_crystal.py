import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- 1. PHYSICAL PARAMETERS (Figure 2c) ---
a_N = 20.0       # Isotropic dynamic nuclear polarization factor (mT)
b_N = 21.0       # Anisotropic DNP factor (mT)
T_N = 0.5        # Nuclear spin relaxation time (s)
B_x = -1.0       # Transverse magnetic field (mT)
B_z = 0.1      # Longitudinal magnetic field for alpha = 10 deg (mT)

B_ext = np.array([B_x, 0.0, B_z])
h = B_ext / np.linalg.norm(B_ext)  # Unit vector of the external magnetic field

# The dynamic nuclear polarization tensor aligned with the external field
alpha_tensor = a_N * np.eye(3) + b_N * np.outer(h, h)

# Electron scaling factor (k_scale = mu_B * g * T_s / hbar)
# Using -60.0 as a robust parameter to mathematically force the Hopf bifurcation
k_scale = 0.7  

# Continuous pump polarization
S0_max = 0.25    
S0 = np.array([0.0, 0.0, S0_max])

# --- 2. STATIONARY BLOCH EQUATION ---
def calc_electron_spin(B_N_flat):
    """Algebraically solves for the electron spin S given the current Overhauser field."""
    K = k_scale * (B_ext + B_N_flat)
    
    # Skew-symmetric matrix for the cross product (K x S)
    M_K = np.array([
        [ 0.0,  -K[2],  K[1]],
        [ K[2],  0.0,  -K[0]],
        [-K[1],  K[0],  0.0 ]
    ])
    
    # Solve (I - M_K) * S = S0
    I = np.eye(3)
    S = np.linalg.solve(I - M_K, S0)
    return S

# --- 3. DYNAMICAL EQUATION FOR THE OVERHAUSER FIELD ---
def overhauser_derivative(t, B_N_flat):
    """Calculates dBN/dt for the ODE solver."""
    S = calc_electron_spin(B_N_flat)
    
    # The macroscopic dynamical equation
    dBN_dt = -(1.0 / T_N) * (B_N_flat - alpha_tensor @ S)
    return dBN_dt

# --- 4. SIMULATION EXECUTION ---
# We will simulate for 60 seconds
t_span = (0, 600) 
t_eval = np.linspace(0, 600, 3000)

# Initial "kick" to prevent the system from settling perfectly on zero
initial_B_N = np.array([0.3, 0.3, 0.3])

print("Simulating the ENSS Continuous Time Crystal. Using Radau solver...")
# Using the Radau method because the extreme timescale separation makes these equations stiff
solution = solve_ivp(
    overhauser_derivative, 
    t_span, 
    initial_B_N, 
    method='RK45', #RK45? or Radau
    t_eval=t_eval,
    rtol=1e-6, atol=1e-8
)

time = solution.t
B_N_history = solution.y

# Calculate the Z-component of the electron spin (proportional to Faraday Rotation)
S_z_history = np.zeros(len(time))
for i in range(len(time)):
    S = calc_electron_spin(B_N_history[:, i])
    S_z_history[i] = S[2]

# --- 5. POST-PROCESSING & PLOTTING ---
# --- 5. POST-PROCESAMIENTO Y GRÁFICOS ---
# Descartar los primeros 20 segundos de caos transitorio mientras el sistema se estabiliza
transient_cutoff = 20
mask = time > transient_cutoff
t_steady = time[mask]
Sz_steady = S_z_history[mask]

# ── GRÁFICO 1: Dominio del Tiempo (Auto-oscilaciones en forma de "M") ──
plt.figure(figsize=(8, 5))
plt.plot(t_steady, Sz_steady, color='#1f77b4', lw=1.5)
#plt.title("Auto-oscilaciones Simuladas (Dominio del Tiempo)", fontsize=14)
plt.xlabel("Tiempo (s)", fontsize=12)
plt.ylabel(r"Polarización del Espín Electrónico $S_z$", fontsize=12)
plt.xlim(30, 90) # Zoom para ver las formas de onda específicas
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('auto_oscilaciones.pdf', format='pdf')
plt.show()

# ── GRÁFICO 2: Retrato de Fase (El Ciclo Límite) ───────────────────────
plt.figure(figsize=(6, 6)) 
tau = 25 # Paso de retardo (delay)
plt.plot(Sz_steady[:-tau], Sz_steady[tau:], color='black', lw=0.8)
#plt.title("Retrato de Fase (Ciclo Límite)", fontsize=14)
plt.xlabel(r"$S_z(t)$", fontsize=12)
plt.ylabel(r"$S_z(t - \tau)$", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('ciclo_limite.pdf', format='pdf')
plt.show()