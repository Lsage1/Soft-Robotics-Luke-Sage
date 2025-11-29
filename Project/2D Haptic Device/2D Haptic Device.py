import numpy as np
import matplotlib.pyplot as plt
from objfun import objfun
import os
import imageio.v2 as imageio
from GetRestDimensions import GetRestDimensions
from DetectJunctions import DetectJunctions
from PlotGeometry import PlotGeometry
from VertexObj import VertexObj, EdgeObj

vertices = np.array([[0,0], [0.02,0], [.04,0], [.06, 0],[.08, 0], [.1, 0], [0.05, -0.05]])
edgeIndex = np.array([[0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [0,6], [3,6], [4,6]])

vertexObjs = []
edgeObjs = []

for index, vert in enumerate(vertices):
    vertexObjs.append(VertexObj(vert[0], vert[1], index))



for edge in edgeIndex:
    v1 = vertexObjs[edge[0]]
    v2 = vertexObjs[edge[1]]
    edgeObj = EdgeObj(v1, v2)
    edgeObjs.append(edgeObj)

    v1.add_edge(edgeObj)
    v2.add_edge(edgeObj)

print("BREAK")

# Get rest angles for each vertex
for vertex in vertexObjs:
    for edgePair in vertex.edgePairs:
        edge1 = edgePair[0]
        edge2 = edgePair[1]
        vertex_nm1 = edge1.get_other_vertex(vertex)
        vertex_np1 = edge2.get_other_vertex(vertex)
        # Get edge vectors

        BA = np.array(vertex_nm1.rest_coords) - np.array(vertex.rest_coords)
        BC = np.array(vertex_np1.rest_coords) - np.array(vertex.rest_coords)

        # Normalize (not required)
        BA_norm = BA / np.linalg.norm(BA)
        BC_norm = BC / np.linalg.norm(BC)

        # Dot and 2D cross product
        dot = np.dot(BA_norm, BC_norm)
        cross = BA_norm[0]*BC_norm[1] - BA_norm[1]*BC_norm[0]

        # Calculate signed Angle
        angle = np.arctan2(cross, dot)
        vertex.rest_angles.append(angle)

# Calculate the rest length of each edge.
for edge in edgeObjs:
    v1 = np.array(edge.vertex1.rest_coords)
    v2 = np.array(edge.vertex2.rest_coords)
    restLength = np.linalg.norm([v2-v1])
    #print(v1, v2, "REST LENGTH", restLength)
    edge.rest_length = restLength

nv = len(vertexObjs) # number of nodes/vertices
ndof = 2 * nv

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
totalTime = 5 # second

# Variables related to plotting
saveImage = 0
plotStep = 5 # Every 5-th step will be plotted

##### UTILITY QUANTITIES

# Utility quantities
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
for k in range(0, nv):
  C[2*k, 2*k] = 6.0 * np.pi * visc * R[k] # Damping along x for k-th node
  C[2*k+1, 2*k+1] = 6.0 * np.pi * visc * R[k] # Damping along y for k-th node

# Initial conditions
q0 = np.zeros(2 * nv)


for c, vertex in enumerate(vertexObjs):
  q0[2*c] = vertex.rest_coords[0]
  q0[2*c+1] = vertex.rest_coords[1]

print(q0)
u0 = np.zeros(2 * nv) # old velocity



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
                        vertexObjs, edgeObjs, free_index)
  if error < 0:
    print('Could not converge.')
    break

  u_new = (q_new - q0) / dt # New velocity


  ctime += dt # Update current time



  q0 = q_new.copy() # New position becomes old position
  u0 = u_new.copy() # New velocity becomes old velocity

  # Plot
  if timeStep % plotStep == 1:
    print("plotting")
    x_arr = q_new[::2] # q[0], q[2], q[4]
    y_arr = q_new[1::2] # q[1], q[3], q[5]

    h1 = plt.figure(1)

    PlotGeometry(q0, edgeIndex)
    plt.title(f't={ctime:.4f}s')


    # Save frame as image
    frame_path = os.path.join(image_folder, f"frame_{timeStep:05d}.png")
    plt.savefig(frame_path, dpi=150)
    frame_files.append(frame_path)
    plt.close()

# --- Assemble video from saved frames ---
video_path = "simulation.mp4"
with imageio.get_writer(video_path, fps=20) as writer:
    for filename in frame_files:
        writer.append_data(imageio.imread(filename))

print(f"Video saved to {video_path}")