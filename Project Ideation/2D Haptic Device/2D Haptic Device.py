import numpy as np
import matplotlib.pyplot as plt
from objfun import objfun
import os
import imageio.v2 as imageio
from GetRestDimensions import GetRestDimensions
from DetectJunctions import DetectJunctions
from PlotGeometry import PlotGeometry

vertices = np.array([[0,0], [0.02,0], [.04,0], [.06, 0],[.08, 0], [.1, 0], [0.05, -0.05]])
edgeIndex = np.array([[0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [0,6], [3,6]])
PlotGeometry(vertices, edgeIndex)
nv = vertices.shape[0] # number of nodes/vertices
ndof = 2 * nv
midNode = nv//2 + 1

# Time step
dt = 0.01 # second

# Rod length
RodLength = 0.1 # meter **** CHANGE

# Discrete length / reference length
deltaL = RodLength / (nv - 1)   #   CHANGE

# Radii of spheres (given) # CHANGE
R = np.zeros(nv)
for k in range(nv):
  R[k] = deltaL/10 # meter
R[midNode-1] = 0.025 # meter

# Densities
rho_metal = 1100 # kg/m^3
rho_gl = 1000
rho = rho_metal  - rho_gl

# Cross-sectional radius
r0 = 0.001 # meter

# Young's modulus
Y = 6e8

# Viscosity
visc = 1000.0 # Pa-s

# Maximum number of iterations
maximum_iter = 1000

# Total time
totalTime = 1 # second

# Variables related to plotting
saveImage = 0
plotStep = 5 # Every 5-th step will be plotted

##### UTILITY QUANTITIES

# Utility quantites
EI = Y * np.pi * r0**4 / 4
EA = Y * np.pi * r0**2

# Tolerance
tol = EI / RodLength ** 2 * 1e-3

# Mass vector and matrix
m = np.zeros( 2 * nv )
for k in range(0, nv):
  m[2*k] = 4/3 * np.pi * R[k]**3 * rho_metal # mass of k-th node along x
  m[2*k + 1] = 4/3 * np.pi * R[k]**3 * rho_metal # mass of k-th node along y
mMat = np.diag(m)

# Gravity (external force)
W = np.zeros( 2 * nv)
g = np.array([0, -9.8]) # m/s^2
for k in range(0, nv):
  W[2*k] = 4.0 / 3.0 * np.pi * R[k]**3 * rho * g[0] # Weight along x
  W[2*k+1] = 4.0 / 3.0 * np.pi * R[k]**3 * rho * g[1] # Weight along y
# Gradient of W = 0

# Viscous damping (external force) *** NO VISCOUS DAMPING
C = np.zeros((2 * nv, 2 * nv))
#for k in range(0, nv):
#  C[2*k, 2*k] = 6.0 * np.pi * visc * R[k] # Damping along x for k-th node
#  C[2*k+1, 2*k+1] = 6.0 * np.pi * visc * R[k] # Damping along y for k-th node

# Initial conditions
q0 = np.zeros(2 * nv)

for c in range(nv):
  q0[2*c] = vertices[c, 0] # x coordinate
  q0[2*c+1] = vertices[c, 1] # y coordinate

junctions = DetectJunctions(vertices, edgeIndex)
print("Junctions:", junctions)

GetRestDimensions(vertices, edgeIndex)


u0 = np.zeros(2 * nv) # old velocity

exit()

##### BOUNDARY CONDITIONS

all_DOFs = np.arange(ndof) # Set of all DOFs
fixed_index = np.array([0, 1, 2, 3]) # Fixed DOFs

# Free index
free_index = np.setdiff1d(all_DOFs, fixed_index) # All the DOFs are free except the fixed ones

### LOOP

# Number of steps
Nsteps = round( totalTime / dt )

ctime = 0 # Current time

# Store the y-coordinate of the middle node, its velocity, and the angle
all_pos = np.zeros(Nsteps)
all_vel = np.zeros(Nsteps)
mid_angle = np.zeros(Nsteps)

all_pos[0] = 0
all_vel[0] = 0
mid_angle[0] = 0

#  Create image folder if none exists
image_folder = "images"
os.makedirs(image_folder, exist_ok=True)
frame_files = []

# Loop over the time steps
for timeStep in range(1,Nsteps):
  print(timeStep)
  q_new, error = objfun(q0, u0, dt, tol, maximum_iter, m, mMat, EI, EA, W, C,
                        deltaL, free_index, vertices)
  if error < 0:
    print('Could not converge.')
    break

  u_new = (q_new - q0) / dt # New velocity


  ctime += dt # Update current time

  # Save information about the middle node
  all_pos[timeStep] = q_new[2*midNode-1] # y coordinate of middle node
  all_vel[timeStep] = u_new[2*midNode-1] # y velocity of middle node
  vec1 = np.array( [ q_new[2*midNode-2], q_new[2*midNode-1], 0 ] ) - np.array( [ q_new[2*midNode-4], q_new[2*midNode-3], 0 ] ) # Second node - First node
  vec2 = np.array( [ q_new[2*midNode], q_new[2*midNode+1], 0 ] ) - np.array( [ q_new[2*midNode-2], q_new[2*midNode-1], 0 ] ) # Third node - Second node
  mid_angle[timeStep] =  np.degrees( np.arctan2(np.linalg.norm( np.cross(vec1, vec2)), np.dot(vec1, vec2)) )

  q0 = q_new.copy() # New position becomes old position
  u0 = u_new.copy() # New velocity becomes old velocity

  # Plot
  if timeStep % plotStep == 1:
    print("plotting")
    x_arr = q_new[::2] # q[0], q[2], q[4]
    y_arr = q_new[1::2] # q[1], q[3], q[5]

    h1 = plt.figure(1)
    plt.clf() # Clear current figure
    plt.plot(x_arr, y_arr, 'ko-')
    plt.title(f't={ctime:.4f}s')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')

    # Save frame as image
    frame_path = os.path.join(image_folder, f"frame_{timeStep:05d}.png")
    plt.savefig(frame_path, dpi=150)
    frame_files.append(frame_path)

# --- Assemble video from saved frames ---
video_path = "simulation.mp4"
with imageio.get_writer(video_path, fps=20) as writer:
    for filename in frame_files:
        writer.append_data(imageio.imread(filename))

print(f"Video saved to {video_path}")