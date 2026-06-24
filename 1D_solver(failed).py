import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- 1. PHYSICAL PARAMETERS ---
a_N, b_N, T_N, B_x, B_z = 20.0, 21.0, 0.5, -1.0, 0.1
B_ext = np.array([B_x, 0.0, B_z])
h = B_ext / np.linalg.norm(B_ext)
alpha_tensor = a_N * np.eye(3) + b_N * np.outer(h, h)

k_scale = 0.7    

# --- 2. SPATIAL GRID (1D) ---
N = 41                 # Number of spatial points
x = np.linspace(-5, 5, N) # Space from -5 um to +5 um
dx = x[1] - x[0]
D = 1.0                # Nuclear Spin Diffusion coefficient (um^2 / s)

# Gaussian Laser Pump: Strong in the center, dark at the edges
S0_max = 0.26          
sigma = 1.5            # Laser beam width (standard deviation)
S0_z = S0_max * np.exp(-x**2 / (2 * sigma**2))

# Store the static spatial pump in a 3xN array
S0_array = np.zeros((3, N))
S0_array[2, :] = S0_z  

# --- 3. THE 1D PDE DERIVATIVE ---
def pde_derivative(t, BN_flat):
    # Reshape the flat 1D array from the solver back into a 3xN spatial grid
    B_N = BN_flat.reshape((3, N))
    
    # 1. Calculate Electron Spin S(x) at every spatial point
    S = np.zeros((3, N))
    I = np.eye(3)
    for i in range(N):
        K_i = k_scale * (B_ext + B_N[:, i])
        M_K = np.array([
            [ 0.0,  -K_i[2],  K_i[1]],
            [ K_i[2],  0.0,  -K_i[0]],
            [-K_i[1],  K_i[0],  0.0 ]
        ])
        S[:, i] = np.linalg.solve(I - M_K, S0_array[:, i])
        
    # 2. Calculate Local Nuclear Dynamics
    local_dynamics = -(1.0 / T_N) * (B_N - alpha_tensor @ S)
    
    # 3. Calculate Spatial Diffusion (Discrete Laplacian)
    diffusion = np.zeros((3, N))
    for j in range(3): # For x, y, and z components
        # Finite difference for second spatial derivative
        diffusion[j, 1:-1] = D * (B_N[j, 2:] - 2*B_N[j, 1:-1] + B_N[j, :-2]) / dx**2
        
        # Neumann boundary conditions (No magnetic flux escaping the edges)
        diffusion[j, 0] = D * (B_N[j, 1] - B_N[j, 0]) / dx**2
        diffusion[j, -1] = D * (B_N[j, -2] - B_N[j, -1]) / dx**2
        
    # Combine local time dynamics and spatial diffusion
    dBN_dt = local_dynamics + diffusion
    
    # Flatten back to a 1D array for scipy's Radau solver
    return dBN_dt.flatten()

# --- 4. EXECUTION ---
t_span = (0, 30) # Simulate 30 seconds of real time
t_eval = np.linspace(0, 30, 600)

# Start with a tiny random magnetization across the grid to allow instability
initial_BN = np.ones((3, N)) * 0.01 

print(f"Solving 1D PDE with {N} spatial points... (Please wait 1-2 minutes)")
sol = solve_ivp(
    pde_derivative, 
    t_span, 
    initial_BN.flatten(), 
    method='Radau', 
    t_eval=t_eval,
    rtol=1e-4, atol=1e-6
)

# --- 5. RECONSTRUCT & PLOT SPATIOTEMPORAL DATA ---
time_steps = len(sol.t)
Sz_history = np.zeros((N, time_steps))

# Re-calculate the electron spin Z-component for plotting
I = np.eye(3)
for t_idx in range(time_steps):
    B_N_current = sol.y[:, t_idx].reshape((3, N))
    for i in range(N):
        K_i = k_scale * (B_ext + B_N_current[:, i])
        M_K = np.array([
            [ 0.0,  -K_i[2],  K_i[1]],
            [ K_i[2],  0.0,  -K_i[0]],
            [-K_i[1],  K_i[0],  0.0 ]
        ])
        S = np.linalg.solve(I - M_K, S0_array[:, i])
        Sz_history[i, t_idx] = S[2]

# Plot the Heatmap
plt.figure(figsize=(12, 6))
plt.contourf(sol.t, x, Sz_history, levels=50, cmap='magma')
cbar = plt.colorbar(label='Electron Spin Polarization $S_z$')
plt.title("1D Spatial Time Crystal: Synchronization Waves", fontsize=15)
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel(r"Distance from Laser Center $x$ ($\mu$m)", fontsize=12)
plt.axhline(0, color='white', linestyle='--', alpha=0.3) # Mark the center
plt.tight_layout()
plt.show()