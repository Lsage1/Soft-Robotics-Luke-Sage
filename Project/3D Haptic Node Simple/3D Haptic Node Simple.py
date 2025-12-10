import os
from getKappa_Junction import getKappa_Junction
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from plotrod_simple import plotrod_simple
from computeTangent import computeTangent
from computeSpaceParallel import computeSpaceParallel
from computeMaterialDirectors import computeMaterialDirectors
from getKappa import getKappa
from objfun import objfun
from computeKappa import computeKappa
from objfun_test import objfun_test

verts = [
    np.array([[0,0,0], [0,0.05,0], [0,0.1,-0.05], [0,0.1,-0.10]]),
    np.array([[0,0,0], [0,-0.05,0], [0,-0.1,-0.05], [0,-0.1,-0.10]]),
    np.array([[0,0,0], [0.05,0,0], [0.1,0,-0.05], [0.1,0,-0.10]])
]

nRods = len(verts)

# Pre-allocate arrays per rod
nv     = [len(v)        for v in verts]
ne     = [len(v)-1      for v in verts]
ndof   = [3*nv[r] + ne[r] for r in range(nRods)]  # trimmed last theta

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

# ---------------- MAIN PER-ROD LOOP ----------------
for r in range(nRods):

    # Build q from verts
    for c in range(nv[r]):
        q[r][4*c:4*c+3] = verts[r][c]
        if 4*c+3 < ndof[r]:
            q[r][4*c+3] = 0.0  # theta (skip last if beyond ndof)

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

    # Initial a1 direction using parallel transport from the junction
    t0 = [0,-1,0]  # junction tangent
    a1_root = [1,0,0]
    t1 = tangent[r][0]  # first tangent of branch r

    # Project parent a1_root into the plane perpendicular to t1
    a1_first = a1_root - np.dot(a1_root, t1) * t1

    # Normalize, with fallback for degeneracy
    n = np.linalg.norm(a1_first)
    if n < 1e-3:
        # fallback if a1_root was almost parallel to t1
        arb = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(arb, t1)) > 0.9:
            arb = np.array([0.0, 1.0, 0.0])
        a1_first = np.cross(t1, arb)
        a1_first /= np.linalg.norm(a1_first)
    else:
        a1_first /= n

    # Space parallel transport along the branch (your existing function)
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

# Get curvature between junction nodes:
kappa_junction = getKappa_Junction(q, m1, m2)

###################################################################


# ---------------- FIXED/FREE DOFS ----------------
fixedIndex = []
freeIndex  = []

for r in range(nRods):
    last_node = nv[r] - 1
    fi = [4 * last_node + i for i in range(3)]  # fix x,y,z of last node
    all_indices = np.arange(ndof[r])
    fr = np.setdiff1d(all_indices, fi)
    fixedIndex.append(fi)
    freeIndex.append(fr)



# ---------------- MASS MATRIX ----------------
rho = 1200
r0 = 0.001
L = np.sum(np.linalg.norm(verts[0][1:] - verts[0][:-1], axis=1))
totalM = L * np.pi * r0**2 * rho
dm = totalM / ne[0]

massVector = np.zeros(ndof[0])
for c in range(nv[0]):
    ind = [4*c, 4*c+1, 4*c+2]
    if c==0 or c==nv[0]-1:
        massVector[ind] = dm/2
    else:
        massVector[ind] = dm
for c in range(ne[0]):
    if 4*c+3 < ndof[0]:
        massVector[4*c+3] = 0.5 * dm * r0**2



massMatrix = np.diag(massVector)

# ---------------- EXTERNAL FORCES ----------------
F_control = [0, 0, 0]
Fg = [np.zeros(ndof[0]) for _ in range(nRods)]
for r in range(nRods):
    last_node = nv[r]-3
    ind = [4*last_node, 4*last_node+1, 4*last_node+2]
    Fg[r][ind] = [0,0,-F_control[r]]

# ---------------- ELASTIC PARAMETERS ----------------
Y = 7e6
nu = 0.5
G = Y/(2*(1+nu))
EA = Y*np.pi*r0**2
EI = Y*np.pi*r0**4/4
GJ = G*np.pi*r0**4/2
dt = 0.01
totalTime = 3
tol = EI / L**2 * 1e-3

# ---------------- REFERENCE TWIST ----------------
refTwist = [np.zeros(nv[r]) for r in range(nRods)]

# ---------------- TIME STEPPING ----------------
Nsteps = round(totalTime/dt)
ctime = 0

a1_old = [a1[r].copy() for r in range(nRods)]
a2_old = [a2[r].copy() for r in range(nRods)]

plotrod_simple(qOld[0], qOld[1], qOld[2], 0)

print("enteringLoop")
# ########################### SQUISH Q vector into 1 Q_global
numRods = len(q)

#### VISCOSITY
visc = 0


q_junction = qOld[0][:3]
q[0] = qOld[0][3:]  # Remove the first 3 elements of q[1] because they will be shared with q[0] at the junction
q[1] = qOld[1][3:]
q[2] = qOld[2][3:]
q_all_old = np.concatenate([q_junction, q[0], q[1], q[2]])

qindex = []
jun_index = np.arange(0,3)
qindex.append(np.arange(len(jun_index), len(q[0])+len(jun_index))) # Get the indices of the first vector
qindex.append(np.arange(len(jun_index)+len(qindex[0]), len(q[0])+len(q[1])+len(jun_index)))
qindex.append(np.arange(len(jun_index)+len(qindex[0])+len(qindex[1]), len(qindex[0])+len(qindex[1]) + len(q[2])+len(jun_index)))

# Get squished free index
fixed_index = [12, 13, 14] #, 24, 25, 26]#, 36, 37, 38]

free_all_index = np.setdiff1d(np.arange(len(q_all_old)), fixed_index)

print(q_all_old)

# Get Squished force index
force_index = [6, 18, 30]
force_amounts = [0.001, 0.001, 0.001]
ext_force_flat = np.zeros(len(q_all_old))
ext_force_flat[force_index] = force_amounts

# Get Squished Mass Vector: NOTE ASSUMES ALL BRANCHES HAVE THE SAME MASS
mass_flat = np.concatenate([massVector[:3], massVector[3:], massVector[3:], massVector[3:]])
mass_matrix_flat = np.diag(mass_flat)

#  Create image folder if none exists
image_folder = "images"
os.makedirs(image_folder, exist_ok=True)
frame_files = []

for timeStep in range(Nsteps):

    # This will work by passing in qOld, uOld, which will be unpacked into q_all_old in objfun.
    # Then, objfun will reassemble qNew from q_all_new and return qNew

    q_new, u_new, a1_new, a2_new = objfun(
        qOld, uOld, qindex, jun_index, a1_old, a2_old,
        free_all_index, dt, tol, refTwist,
        mass_flat, mass_matrix_flat,
        EA, refLen,
        EI, GJ, vorRefLen,
        kappaBar, kappa_junction, twistBar,
        ext_force_flat, nRods, tangent, nv[0], visc
    )

    # Plot the shell
    if timeStep % 2 == 1:
        plotrod_simple(qOld[0], qOld[1], qOld[2], ctime)
        # Save frame as image
        frame_path = os.path.join(image_folder, f"frame_{timeStep:05d}.png")
        plt.savefig(frame_path, dpi=150)
        frame_files.append(frame_path)
        plt.show()

    ctime += dt
    qOld = [q.copy() for q in q_new]
    uOld = [u.copy() for u in u_new]
    a1_old = [a.copy() for a in a1_new]
    a2_old = [a.copy() for a in a2_new]


# --- Assemble video from saved frames ---
video_path = "simulation.mp4"
with imageio.get_writer(video_path, fps=20) as writer:
    for filename in frame_files:
        writer.append_data(imageio.imread(filename))

print(f"Video saved to {video_path}")