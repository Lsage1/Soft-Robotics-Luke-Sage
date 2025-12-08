import numpy as np
import matplotlib.pyplot as plt
from plotrod_simple import plotrod_simple
from computeTangent import computeTangent
from computeSpaceParallel import computeSpaceParallel
from computeMaterialDirectors import computeMaterialDirectors
from getKappa import getKappa
from objfun import objfun
from computeKappa import computeKappa

verts = [
    np.array([[0,0,0], [0,0.05,0], [0,0.1,-0.05], [0,0.1,-0.10]]),
    np.array([[0,0,0], [0,-0.05,0], [0,-0.1,-0.05], [0,-0.1,-0.10]]),
    np.array([[0,0,0], [0.05,0,0], [0.1,0,-0.05], [0.1,0,-0.10]])
]

edges = [
    np.array([[0,1],[1,2],[2,3]]),
    np.array([[0,1],[1,2],[2,3]]),
    np.array([[0,1],[1,2],[2,3]])
]

nRods = len(verts)

# Pre-allocate arrays for each per-rod quantity
nv     = [len(v)        for v in verts]
ne     = [len(v)-1      for v in verts]
ndof   = [4*nvert       for nvert in nv]

q      = [np.zeros(nd)  for nd in ndof]
qOld   = [None]*nRods
uOld   = [None]*nRods

refLen        = [np.zeros(ne[r]) for r in range(nRods)]
vorRefLen     = [np.zeros(nv[r]) for r in range(nRods)]
tangent       = [None]*nRods
a1            = [None]*nRods
a2            = [None]*nRods
m1            = [None]*nRods
m2            = [None]*nRods
kappaBar      = [None]*nRods
twistBar      = [None]*nRods

# -----------------------------------------------------------------------------
# MAIN PER-ROD LOOP (no dictionaries)
# -----------------------------------------------------------------------------

for r in range(nRods):

    # Build q from verts
    for c in range(nv[r]):
        q[r][4*c:4*c+3] = verts[r][c]
        q[r][4*c+3]     = 0.0

    qOld[r] = q[r].copy()
    uOld[r] = np.zeros_like(q[r])

    # Reference lengths
    for c in range(ne[r]):
        refLen[r][c] = np.linalg.norm(verts[r][c+1] - verts[r][c])

    # Voronoi reference lengths
    for c in range(nv[r]):
        if c == 0:
            vorRefLen[r][c] = 0.5 * refLen[r][0]
        elif c == nv[r]-1:
            vorRefLen[r][c] = 0.5 * refLen[r][c-1]
        else:
            vorRefLen[r][c] = 0.5 * (refLen[r][c-1] + refLen[r][c])

    # Tangents
    tangent[r] = computeTangent(qOld[r])

    # Initial a1 direction
    t0 = tangent[r][0]
    arb = np.array([0,0,-1])
    cross = np.cross(t0, arb)
    if np.linalg.norm(cross) < 1e-3:
        arb = np.array([0,1,0])
        cross = np.cross(t0, arb)
    a1_first = cross / np.linalg.norm(cross)

    # Space parallel transport
    a1[r], a2[r] = computeSpaceParallel(a1_first, qOld[r])

    # Theta edges
    theta_nodes = qOld[r][3::4]
    theta_edges = theta_nodes[:ne[r]]

    # Directors
    m1[r], m2[r] = computeMaterialDirectors(a1[r], a2[r], theta_edges)

    # Curvature
    kappaBar[r] = getKappa(qOld[r], m1[r], m2[r])

    # Twist
    twistBar[r] = np.zeros(nv[r])

# Set up boundary conditions: First two nodes and first theta angle is fixed

# Fixed and free DOFs
fixedIndex = []
freeIndex  = []

for r in range(nRods):
    fi = np.arange(0, 7)
    fr = np.arange(7, ndof[r])
    fixedIndex.append(fi)
    freeIndex.append(fr)

segments = verts[1][1:] - verts[1][:-1]
segment_lengths = np.linalg.norm(segments, axis=1)
# Total rod 1 length
L = np.sum(segment_lengths)

r0 = 0.001 # cross-sectional radius of the rod # Given, d = 0.002 m
ctime = 0

# ELASTIC STIFFNESS

# Material Parameters
Y = 7e6 # 10 MPa - Young's modulus
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

# Set up Reference Twist
refTwist = []

for r in range(nRods):     # assumes number of vertices per rod is the same
    refTwist.append(np.zeros(nv[r]))

# MASS VECTORS AND MATRIX

rho = 1200 # kg/m^3 -- density
totalM = L * np.pi * r0**2 * rho  # Total mass of the rod
dm = totalM / ne[0]

massVector = np.zeros(ndof[0])

# NOTE: AGAIN, same mass vector is used for all rods, which assumes that rods are the same length
for c in range(nv[0]): # NOTE: Currently assumes all rods are the same length
  ind = [4*c, 4*c+1, 4*c+2] # x, y, z coordinates of c-th node
  if c == 0 or c == nv[0] - 1:
    massVector[ind] = dm / 2
  else:
    massVector[ind] = dm

print("Ne: ", int(ne[0]))

for c in range(ne[0]):
  massVector[4*c+3] = 0.5 * dm * r0 ** 2 # Equation for a solid cylinder
  # Because r0 is really small, we may get away with just using 0 angular mass

massMatrix = np.diag(massVector)

# External Force: Point load on the last node (instead of gravity)

F_control = 0.1

vectorLoad = np.array([0, 0, -F_control]) # Point load vector

Fg = np.zeros(ndof[0]) # External force vector
c = nv[0]-1
ind = [4*c, 4*c + 1, 4*c + 2] # last node
Fg[ind] += vectorLoad

###################################################################################################
# TIME STEPPING LOOP

Nsteps = round(totalTime / dt ) # number of steps
ctime = 0 # Current time

a1_old = []
a2_old = []

for r in range(nRods):
    a1_old.append(a1[r].copy())
    a2_old.append(a2[r].copy())

# Time frame for considering steady state
track_time = 2 # seconds




# PART 1
for timeStep in range(Nsteps):

  quit()
  q_new, u_new, a1_new, a2_new = objfun(qOld, uOld, a1_old, a2_old,
                                        freeIndex, dt, tol, refTwist,
                                        massVector, massMatrix,
                                        EA, refLen,
                                        EI, GJ, vorRefLen,
                                        kappaBar, twistBar,
                                        Fg)

  quit()









  if timeStep % 10 == 0:
    plotrod_simple(q_new, ctime)

  ctime += dt # Current time
  # Old parameters become new
  qOld = q_new.copy()
  uOld = u_new.copy()
  a1_old = a1_new.copy()
  a2_old = a2_new.copy()


