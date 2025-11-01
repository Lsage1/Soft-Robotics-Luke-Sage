import numpy as np
import matplotlib.pyplot as plt

from objfun import objfun
from crossMat import crossMat
from getFs import getFs
from gradEb import gradEb
from hessEs import hessEs


nv = 50 # number of nodes/vertices
ndof = 2 * nv
midNode = nv//2 + 1

# Time step
dt = 0.01 # second

# Rod length
rodLength = 1 # meter

# Discrete length / reference length
deltaL = rodLength / (nv - 1)

# Radii of spheres (given)
R = np.zeros(nv)
for k in range(nv):
  R[k] = deltaL/10 # meter
R[midNode-1] = 0.025 # meter

# Densities
rho_metal = 7000 # kg/m^3
rho_gl = 1000
rho = rho_metal  - rho_gl

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
totalTime = 0.5 # second

# Variables related to plotting
saveImage = 0
plotStep = 100 # Every 5-th step will be plotted

######################################################################################################

# Utility quantites
ne = nv - 1 # number of edges
EI = ymod * np.pi * (outerRadius**4 - innerRadius**4) / 4 # Flexural Rigidity = Moment of Interia * Young's Modulus
EA = ymod * np.pi * (outerRadius**2 - innerRadius**2) # Axial Rigidity N/m^3

# Tolerance
tol = EI / rodLength ** 2 * 1e-3

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

loadCount = 20
minLoad = 20
maxLoad = 20000

minY_save = np.zeros(loadCount)
minYEuler_save = np.zeros(loadCount)
loads = np.linspace(minLoad, maxLoad, loadCount)

for i, val in enumerate(loads):
    #print(i, val)
    # Gravity (external force)
    W = np.zeros( 2 * nv) # Weight is set to zero
    P = -val
    loadPos = 0.75 # x position at which to apply load approximately

    loadIdx = round(nv * loadPos/rodLength) - 1 # Get the index of the node closest to the load position
    #print(loadIdx)
    W[loadIdx * 2 + 1] = P # Assign load to the y component of the node at the load position
    #print(W)

    #g = np.array([0, -9.8]) # m/s^2
    #for k in range(0, nv):
    #  W[2*k] = 4.0 / 3.0 * np.pi * R[k]**3 * rho * g[0] # Weight along x
    #  W[2*k+1] = 4.0 / 3.0 * np.pi * R[k]**3 * rho * g[1] # Weight along y
    # Gradient of W = 0


    # Viscous damping (external force)
    C = np.zeros((2 * nv, 2 * nv)) # Viscous Damping is set to zero
    #for k in range(0, nv):
    #  C[2*k, 2*k] = 6.0 * np.pi * visc * R[k] # Damping along x for k-th node
    #  C[2*k+1, 2*k+1] = 6.0 * np.pi * visc * R[k] # Damping along y for k-th node

    # Initial conditions
    q0 = np.zeros(2 * nv)
    for c in range(nv):
      q0[2*c] = nodes[c, 0] # x coordinate
      q0[2*c+1] = nodes[c, 1] # y coordinate

    u0 = np.zeros(2 * nv) # old velocity

    ################################################################################
    # Boundary Conditions:

    all_DOFs = np.arange(ndof) # Set of all DOFs
    fixed_index = np.array([0, 1, nv*2-1]) # Fixed DOFs
    #print(all_DOFs)

    # Free index
    free_index = np.setdiff1d(all_DOFs, fixed_index) # All the DOFs are free except the fixed ones

    #################################################################################

    # Number of steps
    Nsteps = round( totalTime / dt )

    ctime = 0 # Current time

    # Store the y-coordinate of the bottom node
    all_pos = np.zeros(Nsteps)


    all_pos[0] = 0


    # Loop over the time steps
    for timeStep in range(1,Nsteps):

      q_new, error = objfun(q0, u0, dt, tol, maximum_iter, m, mMat, EI, EA, W, C,
                            deltaL, free_index)
      if error < 0:
        print('Could not converge.')
        break

      u_new = (q_new - q0) / dt # New velocity


      ctime += dt # Update current time

      # Save position of the bottom node
      minY = (min(q_new[1::2])) # Get the minimum y component of position at this time step

      all_pos[timeStep] = minY # y coordinate of bottom node



      q0 = q_new.copy() # New position becomes old position
      u0 = u_new.copy() # New velocity becomes old velocity

      ########################################################################
      # Calculate theoretical euler beam theory

      c = min(loadPos, (rodLength - loadPos))
      ymin_T = P * c * (rodLength ** 2 - c ** 2) ** 1.5 / (9 * np.sqrt(3) * EI * rodLength)
      #print("Theoretical Minimum Y value", ymin_T)
      #print("Simulated minimum Y value", all_pos[-1])



      # Plot position over time
      if timeStep % plotStep == 0:
        x_arr = q_new[::2] # q[0], q[2], q[4]
        y_arr = q_new[1::2] # q[1], q[3], q[5]

        h1 = plt.figure(1)
        plt.clf() # Clear current figure
        plt.plot(x_arr, y_arr, 'ko-')
        plt.title(f't={ctime:.4f}s')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.show()

    # Store minimum load conditions
    minYEuler_save[i] = ymin_T
    minY_save[i] = minY

      #############################################################################

plt.figure(2)
plt.plot(loads,minYEuler_save, 'ko-', label='Beam Theory')
plt.plot(loads,minY_save, 'ro-', label = 'Simulation Predicted')
plt.xlabel('Applied Load')
plt.ylabel('Maximum Displacement (m)')
plt.title('Maximum Displacement vs. Applied Load')
plt.legend()
plt.show()

#print(minYEuler_save)
#print(minY_save)

# Middle Node info of final case

# Plot the middle node information as a function of time
t_arr = np.linspace(0, totalTime, Nsteps)

plt.figure(3)
plt.plot(t_arr, all_pos, 'ko-')
plt.xlabel('Time (s)')
plt.ylabel('Bottom Node Position (m)')
plt.title('Bottom Node Position vs. Time')
plt.show()
if saveImage:
    plt.savefig('bottom_node_position.png')


