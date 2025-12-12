import numpy as np
import matplotlib.pyplot as plt
from plotrod_simple import plotrod_simple
from computeTangent import computeTangent
from computeSpaceParallel import computeSpaceParallel
from computeMaterialDirectors import computeMaterialDirectors
from getKappa import getKappa
from objfun import objfun
from computeKappa import computeKappa

print("Running Task 1")

# Inputs
nv = 5 # number of nodes
ne = nv - 1
ndof = 3*nv + ne

# Helix parameters
r0 = 0.0015 # cross-sectional radius of the rod # Given, d = 0.002 m
D = 0.04 # meter: helix diameter
pitch = 2 * r0 # Pitch is the same as the cross-sectional diameter
N = 5 # Number of turns
# a and b are parameters used in standard (wikipedia) definition of helix
a = D/2 # Helix radius
b = pitch / (2.0 * np.pi)
T = 2.0 * np.pi * N # Angle created by the helix (N turns in the center)
L = T * np.sqrt( a**2 + b ** 2) # Arc length of the helix
axial_l = N * pitch # Axial length

#print('Helix diameter = ', D)
#print('Pitch = ', pitch)
#rint('N = ', N)
#print('Arc length = ', L)
Estimated_Arc = np.pi * D * N
#print('Estimated arc length = ', Estimated_Arc)
#print('axial_l = ', axial_l)

# Create our nodes matrix
nodes = np.zeros((nv, 3))
for c in range(nv):
  t = c * T / (nv - 1.0)
  nodes[c,0] = a * np.cos(t)
  nodes[c,1] = a * np.sin(t)
  nodes[c,2] = - b * t

nodes = np.array([[0,0,0], [0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0], [0.04, 0, 0]])

# ELASTIC STIFFNESS

# Material Parameters
Y = 7e7 # 10 MPa - Young's modulus
nu = 0.5 # Poisson's ration
G = Y / ( 2 * (1 + nu)) # Shear modulus

# Stiffness variables
EA = Y * np.pi * r0**2 # Stretching stiffness
EI = Y * np.pi * r0**4 / 4.0 # Bending stiffness
GJ = G * np.pi * r0**4 / 2.0 # Twisting stiffness

# TIME PARAMETERS

totalTime = 1.0 # seconds - total time of the simulation
dt = 0.01 # TIme step size -- may need to be adjusted

# Tolerance
tol = EI / L ** 2 * 1e-3

# MASS VECTORS AND MATRIX

rho = 1200 # kg/m^3 -- density
totalM = L * np.pi * r0**2 * rho  # Total mass of the rod
dm = totalM / ne

massVector = np.zeros(ndof)
for c in range(nv):
  ind = [4*c, 4*c+1, 4*c+2] # x, y, z coordinates of c-th node
  if c == 0 or c == nv - 1:
    massVector[ind] = dm / 2
  else:
    massVector[ind] = dm

for c in range(ne):
  massVector[4*c+3] = 0.5 * dm * r0 ** 2 # Equation for a solid cylinder
  # Because r0 is really small, we may get away with just using 0 angular mass

massMatrix = np.diag(massVector)

# External Force: Point load on the last node (instead of gravity)

F_end = EI / L ** 2
vectorLoad = np.array([0, 0, -0.1]) # Point load vector

Fg = np.zeros(ndof) # Eexternal force vector
c = nv-1
ind = [4*c, 4*c + 1, 4*c + 2] # last node
Fg[ind] += vectorLoad

# INITIAL DOF VECTOR

qOld = np.zeros(ndof)
for c in range(nv):
  ind = [4*c, 4*c + 1, 4*c + 2] # c-th node
  qOld[ind] = nodes[c, :]

uOld = np.zeros_like(qOld) # Velocity is zero initially

plotrod_simple(qOld, 0)

# COMPUTE THE REFERENCE LENGTHS:

# Reference length of each edge
refLen = np.zeros(ne)
for c in range(ne):
  refLen[c] = np.linalg.norm(nodes[c + 1, :] - nodes[c, :])

voronoiRefLen = np.zeros(nv)
for c in range(nv):
  if c == 0:
    voronoiRefLen[c] = 0.5 * refLen[c]
  elif c == nv - 1:
    voronoiRefLen[c] = 0.5 * refLen[c - 1]
  else:
    voronoiRefLen[c] = 0.5 * (refLen[c - 1] + refLen[c])

# COMPUTE THE FRAMES

# Reference frame (At t=0, we initialize it with space parallel reference frame but not mandatory)
tangent = computeTangent(qOld)

t0 = tangent[0, :]
arb_v = np.array([0, 0, -1])
a1_first = np.cross(t0, arb_v) / np.linalg.norm(np.cross(t0, arb_v))
if np.linalg.norm(np.cross(t0, arb_v)) < 1e-3: # Check if t0 and arb_v are parallel
  arb_v = np.array([0, 1, 0])
  a1_first = np.cross(t0, arb_v) / np.linalg.norm(np.cross(t0, arb_v))

a1, a2 = computeSpaceParallel(a1_first, qOld)

# Material frame
theta = qOld[3::4] # Extract theta angles
m1, m2 = computeMaterialDirectors(a1, a2, theta)

# NATURAL CURVATURE AND TWIST

# Reference twist
refTwist = np.zeros(nv) # Or use the function we computed

# Natural curvature
kappaBar = getKappa(qOld, m1, m2)

# Natural twist
twistBar = np.zeros(nv)

# Set up boundary conditions: First two nodes and first theta angle is fixed

# Fixed and free DOFs
fixedIndex = np.arange(0, 7)
freeIndex = np.arange(7, ndof)
# If we include the x and y coordinates of the last node as FIXED DOFs, we will get better agreement

###################################################################################################
# TIME STEPPING LOOP

Nsteps = round(totalTime / dt ) # number of steps
ctime = 0 # Current time
endZ_0 = qOld[-1] # End Z coordinate of the first time step
endZ = np.zeros(Nsteps)

a1_old = a1
a2_old = a2

# Time frame for considering steady state
track_time = 2 # seconds
track_steps = round(track_time / dt) # Number of steps to track if steady state has occurred.
zdiff_list =  np.zeros(track_steps)
breakStep = Nsteps # Code will plot until breakStep, but if we reach the final time step, it will plot the whole time.

# PART 1
for timeStep in range(Nsteps):

  q_new, u_new, a1_new, a2_new = objfun(qOld, uOld, a1_old, a2_old,
                                        freeIndex, dt, tol, refTwist,
                                        massVector, massMatrix,
                                        EA, refLen,
                                        EI, GJ, voronoiRefLen,
                                        kappaBar, twistBar,
                                        Fg)

  # Save endZ (z coordinate of the last node)
  endZ[timeStep] = q_new[-1] - endZ_0

  satisfied = False







  if timeStep % 25 == 0:
    plotrod_simple(q_new, ctime)

  ctime += dt # Current time
  # Old parameters become new
  qOld = q_new.copy()
  uOld = u_new.copy()
  a1_old = a1_new.copy()
  a2_old = a2_new.copy()

plt.figure(2)
time_array = np.arange(1, Nsteps+1, 1) * dt
# Plot only until the convergence




print("Finished Task 1")
