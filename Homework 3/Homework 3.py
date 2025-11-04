import numpy as np
import matplotlib.pyplot as plt
import os

from objfun import objfun
from crossMat import crossMat
from getFs import getFs
from gradEb import gradEb
from hessEs import hessEs
from staticSolver import staticSolver
from getDirichlet import getDirichlet

save_folder = "plots"
os.makedirs(save_folder, exist_ok=True)


nv = 51 # number of nodes/vertices
ndof = 2 * nv
midNode = nv//2 + 1

# Time step
dt = 2 # second

# Rod length
rodLength = 1 # meter

# Discrete length / reference length
deltaL = rodLength / (nv - 1)

# Radii of spheres (given)
R = np.zeros(nv)
for k in range(nv):
  R[k] = deltaL/10 # meter
R[midNode-1] = 0.025 # meter



# Cross-sectional radii
innerRadius = 0.011 #m
outerRadius = 0.013 #m

# Young's modulus
ymod = 7e10 # Pa

# Viscosity
visc = 1000.0 # Pa-s

# Maximum number of iterations
maximum_iter = 1000

# Total time
totalTime = 1000 # second

# Variables related to plotting
saveImage = 0
plotStep = 50 # Every 5-th step will be plotted

######################################################################################################

# Utility quantites
ne = nv - 1 # number of edges
EI = ymod * np.pi * (outerRadius**4 - innerRadius**4) / 4 # Flexural Rigidity = Moment of Interia * Young's Modulus
EA = ymod * np.pi * (outerRadius**2 - innerRadius**2) # Axial Rigidity N/m^3

# Tolerance
tol = rodLength * 1e-6

# Geometry
nodes = np.zeros((nv, 2))
for c in range(nv):
  nodes[c, 0] = c * deltaL # x-coordinate
  nodes[c, 1] = 0.0 # y-coordinate

# Mass vector and matrix
density = 2700 # kg/m^3
nodeMass = np.pi * (outerRadius**2 - innerRadius**2) * rodLength * density / (nv - 1)

m = np.zeros( 2 * nv )
for k in range(0, nv):
  m[2*k] = nodeMass # mass of k-th node along x
  m[2*k + 1] = nodeMass # mass of k-th node along y
mMat = np.diag(m)


W = np.zeros( 2 * nv)
g = np.array([0, -9.8]) # m/s^2

p = 2700 # Density
A = np.pi*outerRadius**2 - np.pi*innerRadius**2
m = p * A * rodLength / (nv - 1)

for k in range(0, nv):
  W[2*k] = m * g[0] # Weight along x
  W[2*k+1] = m * g[1] # Weight along y
# Gradient of W = 0

C = np.zeros((2 * nv, 2 * nv))

# Initial conditions
q0 = np.zeros(2 * nv)
for c in range(nv):
  q0[2*c] = nodes[c, 0] # x coordinate
  q0[2*c+1] = nodes[c, 1] # y coordinate

u0 = np.zeros(2 * nv) # old velocity

################################################################################
# Boundary Conditions:

all_DOFs = np.arange(ndof) # Set of all DOFs
fixed_index = np.array([0, 1, nv*2-4, nv*2-3, nv*2-2, nv*2-1]) # Fixed DOFs
control_index = np.array([nv*2-4, nv*2-3, nv*2-2, nv*2-1]) # indices used to control right side of the bar
# Free index
free_index = np.setdiff1d(all_DOFs, fixed_index) # All the DOFs are free except the fixed ones

#################################################################################

# Number of steps
Nsteps = round( totalTime / dt )

ctime = 0 # Current time

# initialize bc positions
xc = 1
yc = 0
thetac = 0 # np.pi / 2

x_mid_c = 0.5
y_mid_c = 0.0

xc_store = np.zeros(Nsteps)
yc_store = np.zeros(Nsteps)
thetac_store = np.zeros(Nsteps)

x_traj_store = np.zeros(Nsteps)
y_traj_store = np.zeros(Nsteps)

for timeStep in range(0, Nsteps):
    x_traj_store[timeStep] = rodLength / 2 * np.cos(np.pi / 2 * timeStep / 1000)
    y_traj_store[timeStep] = - rodLength / 2 * np.sin(np.pi / 2 * timeStep / 1000)

# Loop over the time steps
for timeStep in range(0,Nsteps):
  print(timeStep)

  # Calculate desired x_mid and y_mid from trajectory
  x_mid_c = rodLength / 2 * np.cos(np.pi/2 * timeStep/1000)
  y_mid_c = - rodLength / 2 * np.sin(np.pi/2 * timeStep/1000)

  xc_new, yc_new, thetac_new = getDirichlet(xc, yc, thetac, x_mid_c, y_mid_c, nv, q0, tol, maximum_iter, EI, EA, W, deltaL, free_index)

  xc_new = np.clip(xc_new, 0, 2)
  yc_new = np.clip(yc_new, -1.5, 1.5)
  thetac_new = np.clip(thetac_new, -np.pi/2, np.pi/2)

  q0[nv * 2 - 2] = xc_new  # x_n
  q0[nv * 2 - 1] = yc_new  # y_n
  q0[nv * 2 - 4] = xc_new - deltaL * np.cos(thetac_new)  # X n-1
  q0[nv * 2 - 3] = yc_new - deltaL * np.sin(thetac_new)  # Y n-1

  q_new, error = objfun(q0, u0, dt, tol, maximum_iter, m, mMat, EI, EA, W, C,
                        deltaL, free_index, fixed_index)
  if error < 0:
    print('Could not converge.')
    break

  u_new = (q_new - q0) / dt # New velocity
  ctime += dt # Update current time
  # Save position of the bottom node
  minY = (min(q_new[1::2])) # Get the minimum y component of position at this time step

  # Store control inputs over time
  xc_store[timeStep] = xc_new
  yc_store[timeStep] = yc_new
  thetac_store[timeStep] = thetac_new

  xc = xc_new.copy()
  yc = yc_new.copy()
  thetac = thetac_new.copy()
  q0 = q_new.copy() # New position becomes old position
  u0 = u_new.copy() # New velocity becomes old velocity

  ########################################################################

  # Plot position over time
  if timeStep % plotStep == 0:
    x_arr = q_new[::2] # q[0], q[2], q[4]
    y_arr = q_new[1::2] # q[1], q[3], q[5]

    h1 = plt.figure(1)
    plt.clf() # Clear current figure
    plt.plot(x_arr, y_arr, 'ko-')
    plt.plot(x_mid_c, y_mid_c, 'ro')
    plt.plot(x_traj_store, y_traj_store, 'r-')
    plt.title(f't={ctime:.4f}s')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.title(f't={ctime:.4f}s')
    filename = f"plot{timeStep:03d}.png"
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")
    #plt.show()


      #############################################################################
h2 = plt.figure(2)
plt.clf()
plt.xlabel('time (sec)')
plt.ylabel('x control position (m)')
plt.plot(np.linspace(0, totalTime, Nsteps), xc_store)
plt.show()


h3 = plt.figure(3)
plt.clf()
plt.xlabel('time (sec)')
plt.ylabel('y control position (m)')
plt.plot(np.linspace(0, totalTime, Nsteps), yc_store)
plt.show()

h4 = plt.figure(4)
plt.clf()
plt.xlabel('time (sec)')
plt.ylabel('theta control position (rad)')
plt.plot(np.linspace(0, totalTime, Nsteps), thetac_store)
plt.show()



