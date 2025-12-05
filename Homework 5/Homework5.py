import numpy as np
import matplotlib.pyplot as plt
from plotShell import plotShell
from getTheta import getTheta
from objfun import objfun
#import os
#import imageio.v2 as imageio



x0 = np.array([
    # ---- Top row (y = 0.01) ----
    [-0.0125, 0.01, 0],   # 0
    [ 0.0000, 0.01, 0],   # 1
    [ 0.0125, 0.01, 0],   # 2
    [ 0.0250, 0.01, 0],   # 3
    [ 0.0375, 0.01, 0],   # 4
    [ 0.0500, 0.01, 0],   # 5
    [ 0.0625, 0.01, 0],   # 6
    [ 0.0750, 0.01, 0],   # 7
    [ 0.0875, 0.01, 0],   # 8
    [ 0.1000, 0.01, 0],   # 9

    # ---- Bottom row (y = 0.00) ----
    [-0.0125, 0.00, 0],   # 10
    [ 0.0000, 0.00, 0],   # 11
    [ 0.0125, 0.00, 0],   # 12
    [ 0.0250, 0.00, 0],   # 13
    [ 0.0375, 0.00, 0],   # 14
    [ 0.0500, 0.00, 0],   # 15
    [ 0.0625, 0.00, 0],   # 16
    [ 0.0750, 0.00, 0],   # 17
    [ 0.0875, 0.00, 0],   # 18
    [ 0.1000, 0.00, 0],   # 19
])

qOld = x0.flatten()

nv = 20
ndof = 3 * nv
visc = .01 # May need for convergence

# Create edges (stretching) and hinges (bending)
edges = np.array([
    # Vertical edges
    (0,10), (1,11), (2,12), (3,13), (4,14), (5,15),
    (6,16), (7,17), (8,18), (9,19),

    # Top horizontal edges
    (0,1), (1,2), (2,3), (3,4), (4,5),
    (5,6), (6,7), (7,8), (8,9),

    # Bottom horizontal edges
    (10,11), (11,12), (12,13), (13,14), (14,15),
    (15,16), (16,17), (17,18), (18,19),

    # Diagonals
    (0,11), (11,2),
    (13,2), (13,4),
    (15,4), (15,6),
    (17,6), (17,8),
    (19,8)
])


hinges = np.array([
    (0,11,10,1),
    (11,2,1,12),
    (2,13,12,3),
    (13,4,3,14),
    (4,15,14,5),
    (15,6,5,16),
    (6,17,16,7),
    (17,8,7,18),
    (8,19,18,9),

    (1, 11, 0, 2),
    (2, 12, 11, 13),
    (3, 13, 2, 4),
    (4, 14, 13, 15),
    (5, 15, 4, 6),
    (6, 16, 15, 17),
    (7, 17, 6, 8),
    (8, 18, 17, 19)
])

plotShell(qOld, edges, 0)


# Elastic Stiffness
Y = 1.0e7 # Young's modulus in Pa
h = 0.002 # Thickness in meter

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
totalTime = 2 # seconds
dt = 0.001 # time step sie

tol = kb / (0.01) * 1e-3 # Approximate tolerance

# Mass Vector and Matrix:
rho = 1000 # kg/m^3Density
length = .1 # m
width = 0.02 #m

totalM = width * h * length * rho # total mass in kg
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

# Natural Curvature | NOTE: For current setup, curvature of each hinge is zero
thetaBar = np.zeros(hinges.shape[0]) # getTheta(qOld)
for kHinge in range(hinges.shape[0]):
    n0_idx, n1_idx, n2_idx, n3_idx = hinges[kHinge]
    x0 = qOld[3*n0_idx:3*n0_idx+3]
    x1 = qOld[3*n1_idx:3*n1_idx+3]
    x2 = qOld[3*n2_idx:3*n2_idx+3]
    x3 = qOld[3*n3_idx:3*n3_idx+3]
    thetaBar[kHinge] = getTheta(x0, x1, x2, x3)

print(thetaBar)

# Boundary Conditions and Initial Conditions
# qOld is already computed
uOld = np.zeros(ndof) # Initialize velocity

fixed_nodes = [0, 1, 10, 11]  # choose three non-collinear nodes
fixedIndex = np.hstack([3*np.array(fixed_nodes) + i for i in range(3)])
freeIndex = np.setdiff1d(np.arange(ndof), fixedIndex)

#  Create image folder if none exists
#image_folder = "images"
#os.makedirs(image_folder, exist_ok=True)
#frame_files = []

# ######################## TIME STEPPING LOOP ##########################

Nsteps = round(totalTime / dt) # Number of time steps
ctime = 0 # Current time
endZ = np.zeros(Nsteps) # z-coordinate of the last node
print("starting loop")
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
  if timeStep % 1000 == 1:
    plotShell(qOld, edges, ctime)
    # Save frame as image
    #frame_path = os.path.join(image_folder, f"frame_{timeStep:05d}.png")
    #plt.savefig(frame_path, dpi=150)
    #frame_files.append(frame_path)
    plt.show()
    #plt.close()


# --- Assemble video from saved frames ---
#video_path = "simulation.mp4"
#with imageio.get_writer(video_path, fps=20) as writer:
#    for filename in frame_files:
#        writer.append_data(imageio.imread(filename))

#print(f"Video saved to {video_path}")

# Visualize
plt.figure(2)
time_array = np.arange(1, Nsteps+1) * dt
plt.plot(time_array, endZ, 'ro-')
plt.box(True)
plt.xlabel('Time, t [sec]')
plt.ylabel('z-coord of last node, $\\delta_z$ [m]')
plt.show()

plt.figure(3)
q = totalM * -9.81 / length # load / unit length
I = width * h ** 3 / 12
deltaEB_tip = q * length ** 4 / (8 * Y * I)
deltaEB = np.zeros(8)
x_sweep = np.linspace(0, 0.1, 8)
for index, x in enumerate(x_sweep):
    deltaEB[index] = q * 4 / (24 * Y * I) * (x**4 - 4*length*x**3 + 6*length**2*x**2)

plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.plot(x_sweep, deltaEB, "bo-", label="Euler-Bernoulli Beam Theory")
points = qNew.reshape(-1, 3)  # N×3 array
xs = points[:, 0][:10]
zs = points[:, 2][:10]
plt.plot(xs, zs, 'ro-',  label="Discrete Shell Simulation")
plt.show()

