import numpy as np
import matplotlib.pyplot as plt
from plotShell import plotShell
from getTheta import getTheta
from objfun import objfun

x0 = np.array([0, 0, 0])
x1 = np.array([0.01, 0, 0])
x2 = np.array([0.005, 0.01, 0])
x3 = np.array([0.005, -0.01, 0])

qOld = np.concatenate((x0, x1, x2, x3))
plotShell(qOld, 0)

nv = 4
ndof = 3 * nv
visc = 0 # May need for convergence

# Create edges (stretching) and hinges (bending)
edges = np.array( [ (0,1), (0,2), (0,3), (1,2), (1,3)  ] )
hinges = np.array( [ (0,1, 2, 3)  ] )

# Elastic Stiffness
Y = 1.0e7 # Young's modulus in Pa
h = 0.001 # Thickness in meter

# Stiffness variables
kb = 2.0 / np.sqrt(3.0) * Y * h**3.0 / 12 # Bending stiffness (Newton-meter)
refLen = np.zeros(edges.shape[0]) # reference length, denoted as l_k
ks = np.zeros_like(refLen) # Stretching stiffness
for kEdge in range(edges.shape[0]):
  node0 = edges[kEdge, 0]
  node1 = edges[kEdge, 1]
  x0 = qOld[ 3*node0: 3*node0 + 3]
  x1 = qOld[ 3*node1: 3*node1 + 3]
  refLen[kEdge] = np.linalg.norm(x1 - x0)
  ks[kEdge] = np.sqrt(3.0) / 2.0 * Y * h * (refLen[kEdge]) ** 2

# Time Parameters
totalTime = 5 # seconds
dt = 0.001 # time step sie

tol = kb / (0.01) * 1e-3 # Approximate tolerance

# Mass Vector and Matrix:
rho = 10 # Density
totalM = 0.01 # total mass in kg
dm = totalM / nv # mass per nodes -- approximation
massVector = np.zeros(ndof)
for c in range(nv): # Loop over every node
  ind = [3*c, 3*c+1, 3*c+2] # location of the c-th node in the DOF vector
  massVector[ind] = dm

massMatrix = np.diag(massVector)

# External Force:
g = np.array([0, 0, -9.8])
Fg = np.zeros(ndof)
for c in range(nv):
  ind = [3*c, 3*c+1, 3*c+2]
  Fg[ind] = massVector[ind] * g

# Natural Curvature
thetaBar = getTheta(x0, x1, x2, x3) # getTheta(qOld)

# Boundary Conditions and Initial Conditions
# qOld is already computed
uOld = np.zeros(ndof) # Initialize velocity

fixedIndex = np.arange(0,9)
freeIndex = np.arange(9, ndof) # Only x_3 is free

# ######################## TIME STEPPING LOOP ##########################

Nsteps = round(totalTime / dt) # Number of time steps
ctime = 0 # Current time
endZ = np.zeros(Nsteps) # z-coordinate of the last node

for timeStep in range(Nsteps):
  print('Current time =%f' % ctime)

  qNew, uNew = objfun(qOld, uOld, freeIndex, dt, tol, massVector, massMatrix, ks, refLen, edges, kb, thetaBar, hinges, Fg, visc)
  ctime += dt

  # Update my old positions and velocities
  qOld = qNew.copy()
  uOld = uNew.copy()

  # Store endZ
  endZ[timeStep] = qNew[-1]

  # Plot the shell
  if timeStep % 1000 == 0:
    plotShell(qOld, ctime)

# Visualize
plt.figure(2)
time_array = np.arange(1, Nsteps+1) * dt
plt.plot(time_array, endZ, 'ro-')
plt.box(True)
plt.xlabel('Time, t [sec]')
plt.ylabel('z-coord of last node, $\\delta_z$ [m]')
plt.show()