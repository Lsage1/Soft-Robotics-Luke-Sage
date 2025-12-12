import os

import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from plotrod_simple import plotrod_simple
from computeTangent import computeTangent
from computeSpaceParallel import computeSpaceParallel
from computeMaterialDirectors import computeMaterialDirectors
from getKappa import getKappa
from objfun import objfun
from PlotRodNetwork import PlotRodNetwork
from GeometryObjects import EdgeObj
from GeometryObjects import VertexObj
from ComputeTangentEdges import ComputeTangentEdges
from computeSpaceParallel_OO import computeSpaceParallel_OO
from getKappa_OO import getKappa_OO
from computeMaterialDirectors_OO import computeMaterialDirectors_OO
from tree_traverse import tree_CSP_CMD
from tree_traverse import tree_getKappa

#vertices = np.array([[0,0,0],
#                     [0,0.05,0], [0.05,0,0], [0,-0.05,0], [-0.05, 0,0],
#                     [0,0.1,-0.05], [0.1,0,-0.05], [0,-0.1,-0.05], [-0.1, 0,-0.05],
#                     [0,0.15,-0.1], [0.15,0,-0.1], [0,-0.15,-0.1], [-0.15, 0,-0.1],
#                     [-0.1, 0, -0.1], [-0.05, 0.05, 0], [-0.1, 0.05, 0]])
#edges = np.array([[0,1], [0,2], [0,3], [0,4], [1,5], [2,6], [3,7], [4,8], [5,9], [6, 10], [7, 11], [8, 12], [8,13], [1,14], [14, 15]])

#vertices = np.array([[0,0,0], [0.01, 0, 0], [0.02, 0, 0], [0, -0.01, 0], [0, -0.02, 0], [0, 0,-0.01], [0, 0, -0.02]])
#edges = np.array([[0,1], [1,2], [0,3], [3, 4], [0, 5], [5,6]])

vertices = np.array([[0,0,0], [0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0], [0.04, 0, 0]])
edges = np.array([[0,1], [1,2], [2,3], [3, 4]])

nv = len(vertices) # number of nodes
ne = len(edges)
ndof_tr = 3*nv # translational degrees of freedom
ndof_rot = ne # rotational degrees of freedom
ndof = ndof_rot + ndof_tr

vertexObjs = []
edgeObjs = []

# Create a list of vertex objects, and create a vertex object for each coordinate set
for index, vert in enumerate(vertices):
    vertexObjs.append(VertexObj(vert[0], vert[1], vert[2]))

# Store a list of edge objects
for edge in edges:
    v1 = vertexObjs[edge[0]] # Get the vertex object associated with the index retrieved
    v2 = vertexObjs[edge[1]]
    edgeObj = EdgeObj(v1, v2, 0) # Create a new edge object with the two vertices that define it. For now, Theta = 0
    edgeObjs.append(edgeObj) # Add new edge to the list of vertexes

    # For both vertexes making up the edge, add a reference to the edge
    v1.add_edge(edgeObj)
    v2.add_edge(edgeObj)

# Get rest angles for each vertex
for vertex in vertexObjs:
    for edgePair in vertex.edgePairs:
        edge1 = edgePair[0]
        edge2 = edgePair[1]
        vertex_nm1 = edge1.get_other_vertex(vertex)
        vertex_np1 = edge2.get_other_vertex(vertex)
        # Get edge vectors

for vertex in vertexObjs:
    print(vertex.edges)

    # Detect Junctions and Rod Ends
    if len(vertex.edges) >= 3:
        vertex.junction = True
    elif len(vertex.edges) == 1:
        vertex.end = True

    # Initialize vertex coordinates at same location as rest coordinates
    vertex.coords = vertex.rest_coords


# Calculate the rest length of each edge.
for edge in edgeObjs:
    v1 = np.array(edge.vertex1.rest_coords)
    v2 = np.array(edge.vertex2.rest_coords)
    restLength = np.linalg.norm(v2-v1)
    edge.rest_length = restLength

# Calculate Voronoi Length for each edge pair
for vertex in vertexObjs:
    for i, edgePair in enumerate(vertex.edgePairs):
        vertex.voronoi_length.append((edgePair[0].rest_length + edgePair[1].rest_length) / 2)

r0 = 0.0015 # rod radius

PlotRodNetwork(vertexObjs, edgeObjs, extra_vertices=None, extra_edges=None)

# ELASTIC STIFFNESS

# Material Parameters
Y = 7e6 # 10 MPa - Young's modulus
nu = 0.5 # Poisson's ration. Standard for elastomers
G = Y / ( 2 * (1 + nu)) # Shear modulus

# Stiffness variables **** NOTE: MAKE THIS VERTEX OR EDGE BASED
EA = Y * np.pi * r0**2 # Stretching stiffness
EI = Y * np.pi * r0**4 / 4.0 * 100 # Bending stiffness
GJ = G * np.pi * r0**4 / 2.0 # Twisting stiffness

# TIME PARAMETERS

totalTime = 2 # seconds - total time of the simulation
dt = 0.01 # TIme step size -- may need to be adjusted

# Tolerance
tol = EI / 0.1 ** 2 * 1e-3  ######################################## CHANGE



# INITIAL DOF VECTORs

qOld_tr = np.zeros(ndof_tr)
qOld_rot = np.zeros(ndof_rot)
uOld_tr = np.zeros_like(qOld_tr) # Velocity is zero initially
# Note: rotational (along edge axis) inertia is not tracked


# COMPUTE THE FRAMES
# Reference frame (At t=0, we initialize it with space parallel reference frame but not mandatory)

# Call function to calculate tangent vector for each edge object
ComputeTangentEdges(edgeObjs, True)
for edge in edgeObjs:
    edge.rest_tangent = edge.tangent
    edge.tangent0 = edge.tangent

print(len(qOld_tr), "Translational Degrees of Freedom")
print(len(qOld_rot), "Rotational Degrees of Freedom")


first_end_vertex = next(v for v in vertexObjs if v.end) # Get the first vertex with an end
end_edge = first_end_vertex.edges[0] # This vertex is an end, so it has one edge, which will be the first in its list
end_edge.network_root = True
t0 = end_edge.tangent


# COMPUTE END MATERIAL DIRECTORS
arb_v = np.array([0, 0, -1]) # At the end node, create a reference vector.
a1_first = np.cross(t0, arb_v) / np.linalg.norm(np.cross(t0, arb_v))
if np.linalg.norm(np.cross(t0, arb_v)) < 1e-3:  # Check if t0 and arb_v are parallel
    arb_v = np.array([0, 1, 0])
    a1_first = np.cross(t0, arb_v) / np.linalg.norm(np.cross(t0, arb_v))
a1_first = a1_first / np.linalg.norm(a1_first)  # Ensure it is unit
end_edge.a1 = a1_first
end_edge.a2 = np.cross(t0, a1_first)
end_edge.a2 = end_edge.a2 / np.linalg.norm(end_edge.a2)  # Ensure it is unit
computeMaterialDirectors_OO(end_edge)

# /\/\/\/\ Tree: Compute Space Parallel and Compute Material Directors /\/\/\/\
end_edge.handled = True
tree_CSP_CMD(end_edge, first_end_vertex, vertexObjs, edgeObjs)

#  /\/\/\/\ TREE GET KAPPA /\/\/\/\
for vertex in vertexObjs: # Initialize a twist vector in every object
    vertex.ref_twist = np.zeros(len(vertex.children))
# DO A SECOND TREE PASS. We need to do this to getKappa for each edge, since we don't yet know m1, m2 for the current edge in the first tree seach.
for e in edgeObjs: # Reset handled flags
    e.handled = False
v0 = first_end_vertex
v1 = end_edge.get_other_vertex(v0)
end_edge.handled = True # Mark starting edge handled
tree_getKappa(end_edge, v1, vertexObjs, edgeObjs)

for vertex in vertexObjs:
    vertex.rest_kappa = vertex.kappa.copy()
    vertex.junction_rest_kappa = [kappa_pair.copy() for kappa_pair in vertex.junction_kappa]


######################### Q_O vector assembly ###########################
# Establish a Q_0 vector from degrees of freedom. It takes the for [x0, y0, z0, ... theta0, theta1, ...]
# Build index maps ONCE
for c, vertex in enumerate(vertexObjs):
    vertex.index = [c*3, c*3+1, c*3+2]

for d, edge in enumerate(edgeObjs):
    edge.theta_index = ndof_tr + d

# Insert vertex positions
q_0 = np.zeros(ndof)
for vertex in vertexObjs:
    q_0[vertex.index] = vertex.coords

# Insert twist DOFs
for edge in edgeObjs:
    q_0[edge.theta_index] = edge.rest_twist
# TIME INTEGRATION LOOP

######################### MASS VECTORS + MATRIX ###############################
# NOTE: still need to handle junction additional mass
rho = 1200 # kg/m^3 -- density
# CHANGE TOTAL LENGTH LATER *********************************************
totalM = 0.05*8 * np.pi * r0**2 * rho  # Total mass of the rod
dm = totalM / ne


massVector = np.zeros(ndof)
for vertex in vertexObjs:
  if vertex.end:
      massVector[vertex.index] = dm/2
  else:
      massVector[vertex.index] = dm
for edge in edgeObjs:
    massVector[edge.theta_index] = 0.5 * dm * r0 ** 2 # Equation for a solid cylinder
  # Because r0 is really small, we may get away with just using 0 angular mass

massMatrix = np.diag(massVector)

# ################### External Force: Point Loads ##################

F_end = 0.1 # CHANGE LATER
vectorLoad = np.array([0, 0, -0.001]) # Point load vector

Fg = np.zeros(ndof) # External force vector
vertexObjs[4].f_ext = vectorLoad
for v in vertexObjs:
    Fg[v.index] += v.f_ext


############### BOUNDARY CONDITIONS ####################
# Set up boundary conditions: Index of fixed degrees of freedom. Form: [[VertexIndex, [X, Y, Z]]...]
vertexObjs[0].fix([True, True, True])
vertexObjs[1].fix([True, True, True])
# Index of edges with fixed rotation. NOTE: Currently no edges have fixed rotation.
edgeObjs[0].twist_fixed = True

 # Assemble free_index vector
ndof_total = len(vertexObjs)*3 + len(edgeObjs) # Total number of DOFS:

free_index = list(range(ndof_total))
for vertex in vertexObjs:
    for i, fixed in enumerate(vertex.dofs_fixed):
        if fixed:
            dof_idx = vertex.index[i]

            if dof_idx in free_index:
                free_index.remove(dof_idx)

# Remove fixed rotational DOFs
for d, edge in enumerate(edgeObjs):
    dof_idx = ndof_tr + d
    if edge.twist_fixed:
        if dof_idx in free_index:
            free_index.remove(dof_idx)

free_index = np.array(free_index, dtype=int)

qOld = q_0
uOld = np.zeros(len(q_0))



Nsteps = round(totalTime / dt ) # number of steps
ctime = 0 # Current time

for edge in edgeObjs:
    edge.a1_old = edge.a1.copy()
    edge.a2_old = edge.a2.copy()
    edge.tangent0 = edge.tangent.copy()

#  Create image folder if none exists
image_folder = "images"
os.makedirs(image_folder, exist_ok=True)
frame_files = []



print("entering Loop")
for timeStep in range(Nsteps):


  q_new, u_new = objfun(end_edge, first_end_vertex, edgeObjs, vertexObjs,
                        qOld, uOld,
                        free_index, dt, tol,
                        massVector, massMatrix,
                        EA, EI, GJ, Fg)


  qOld = q_new.copy()
  uOld = u_new.copy()
  print('Current time: ', ctime, " Idx: ", timeStep)



  ctime += dt # Current time


  if timeStep % 2 == 1:
      PlotRodNetwork(vertexObjs, edgeObjs, extra_vertices=None, extra_edges=None)      # Save frame as image
      frame_path = os.path.join(image_folder, f"frame_{timeStep:05d}.png")
      plt.savefig(frame_path, dpi=150)
      frame_files.append(frame_path)
      plt.close()

  if timeStep % 10 == 1:
      PlotRodNetwork(vertexObjs, edgeObjs, extra_vertices=None, extra_edges=None)      # Save frame as image
      plt.show()

  # Old parameters become new
  for edge in edgeObjs:
      edge.a1_old = edge.a1.copy()
      edge.a2_old = edge.a2.copy()
      edge.tangent0 = edge.tangent.copy()

# --- Assemble video from saved frames ---
video_path = "simulation.mp4"
with imageio.get_writer(video_path, fps=20) as writer:
    for filename in frame_files:
        writer.append_data(imageio.imread(filename))

print(f"Video saved to {video_path}")

PlotRodNetwork(vertexObjs, edgeObjs, extra_vertices=None, extra_edges=None)  # Save frame as image
plt.show()