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


vertices = np.array([[0,0,0],
                     [0,0.05,0], [0.05,0,0], [0,-0.05,0], [-0.05, 0,0],
                     [0,0.1,-0.05], [0.1,0,-0.05], [0,-0.1,-0.05], [-0.1, 0,-0.05],
                     [0,0.15,-0.1], [0.15,0,-0.1], [0,-0.15,-0.1], [-0.15, 0,-0.1],
                     [-0.1, 0, -0.1], [-0.05, 0.05, 0], [-0.1, 0.05, 0]])
edges = np.array([[0,1], [0,2], [0,3], [0,4], [1,5], [2,6], [3,7], [4,8], [5,9], [6, 10], [7, 11], [8, 12], [8,13], [1,14], [14, 15]])

# Inputs
nv = len(vertices) # number of nodes
ne = len(edges)
ndof_tr = 3*nv # translational degrees of freedom
ndof_rot = ne # rotational degrees of freedom

vertexObjs = []
edgeObjs = []

# Create a list of vertex objects, and create a vertex object for each coordinate set
for index, vert in enumerate(vertices):
    vertexObjs.append(VertexObj(vert[0], vert[1], vert[2], index))

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

        BA = -np.array(vertex_nm1.rest_coords) + np.array(vertex.rest_coords)
        BC = np.array(vertex_np1.rest_coords) - np.array(vertex.rest_coords)

        # Normalize (not required)
        BA_norm = BA / np.linalg.norm(BA)
        BC_norm = BC / np.linalg.norm(BC)

        # Dot and cross product
        dot = np.dot(BA_norm, BC_norm)
        cross = np.cross(BA_norm, BC_norm)

        # Calculate signed Angle
        angle = np.arctan2(cross, dot)
        vertex.rest_angles.append(angle)

    # Detect Junctions and Rod Ends
    if len(vertex.edgePairs) > 1:
        vertex.junction = True
    if len(vertex.edgePairs) == 0:
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

r0 = 0.0015 # rest radius

nodes = vertices ### CHANGE LATER

# ELASTIC STIFFNESS

# Material Parameters
Y = 7e7 # 10 MPa - Young's modulus
nu = 0.5 # Poisson's ration. Standard for elastomers
G = Y / ( 2 * (1 + nu)) # Shear modulus

# Stiffness variables
EA = Y * np.pi * r0**2 # Stretching stiffness
EI = Y * np.pi * r0**4 / 4.0 # Bending stiffness
GJ = G * np.pi * r0**4 / 2.0 # Twisting stiffness

# TIME PARAMETERS

totalTime = 1 # seconds - total time of the simulation
dt = 0.01 # TIme step size -- may need to be adjusted

# Tolerance
tol = EI / 0.1 ** 2 * 1e-3  ######################################## CHANGE

# MASS VECTORS AND MATRIX

rho = 1200 # kg/m^3 -- density
# CHANGE TOTAL LENGTH LATER *********************************************
totalM = 0.05*8 * np.pi * r0**2 * rho  # Total mass of the rod
dm = totalM / ne

massVector = np.zeros(ndof_tr)
for c in range(nv):
  ind = [3*c, 3*c+1, 3*c+2] # x, y, z coordinates of c-th node
  if c == 0 or c == nv - 1:
    massVector[ind] = dm / 2
  else:
    massVector[ind] = dm

for c in range(ne):
  massVector[3*c+3] = 0.5 * dm * r0 ** 2 # Equation for a solid cylinder
  # Because r0 is really small, we may get away with just using 0 angular mass

massMatrix = np.diag(massVector)


# External Force: Point load on the last node (instead of gravity)

F_end = 0.0 # CHANGE LATER
vectorLoad = np.array([0, 0, -F_end]) # Point load vector

Fg = np.zeros(ndof_tr) # External force vector
c = 0 # node at which to apply the load
ind = [3*c, 3*c + 1, 3*c + 2] # last node
Fg[ind] += vectorLoad

# INITIAL DOF VECTOR

qOld_tr = np.zeros(ndof_tr)
qOld_rot = np.zeros(ndof_rot)
for c in range(nv):
  ind = [3*c, 3*c + 1, 3*c + 2] # c-th node
  qOld_tr[ind] = nodes[c, :]


uOld_tr = np.zeros_like(qOld_tr) # Velocity is zero initially
# Note: rotational (along edge axis) inertia is not tracked

#PlotRodNetwork(vertexObjs, edgeObjs)

# COMPUTE THE FRAMES
# Reference frame (At t=0, we initialize it with space parallel reference frame but not mandatory)

# Call function to calculate tangent vector for each edge object
ComputeTangentEdges(edgeObjs)

# tangent = computeTangent(qOld_tr)


# NOTE:  For each "Start point" do this. Calculate a reference vector, considered the origin of no rotation
print(len(qOld_tr), "Translational Degrees of Freedom")
print(len(qOld_rot), "Rotational Degrees of Freedom")

# Begin a Network wide sweep. Essentially a tree search, establishing startup stuff


first_end_vertex = next(v for v in vertexObjs if v.end) # Get the first vertex with an end


end_edge = first_end_vertex.edges[0] # This vertex is an end, so it has one edge, which will be the first in its list
# Create an arbitrary vector for the first edge
t0 = end_edge.tangent

# At the end node, create a reference vector.
arb_v = np.array([0, 0, -1])
a1_first = np.cross(t0, arb_v) / np.linalg.norm(np.cross(t0, arb_v))

if np.linalg.norm(np.cross(t0, arb_v)) < 1e-3:  # Check if t0 and arb_v are parallel
    arb_v = np.array([0, 1, 0])
    a1_first = np.cross(t0, arb_v) / np.linalg.norm(np.cross(t0, arb_v))

a1_first = a1_first / np.linalg.norm(a1_first)  # Ensure it is unit

end_edge.a1 = a1_first
end_edge.a2 = np.cross(t0, a1_first)
end_edge.a2 = end_edge.a1 / np.linalg.norm(end_edge.a2)  # Ensure it is unit
computeMaterialDirectors_OO(end_edge)
end_edge.handled = True

# TREE SEARCH

tree_search_active = True
edge01 = end_edge
vertex0 = first_end_vertex
activeJunction = [] # When a junction is reached, store the junction vertex, and the root edge that led into it

## V0 ------E01------ V1 -------E12------- V2
while tree_search_active:
    vertex1 = edge01.get_other_vertex(vertex0)
    # Only add junction if it hasn't been added yet
    if vertex1.junction:
        # Get all outgoing edges except the one we came from
        remaining = [e for e in vertex1.edges if (not e.handled and e is not edge01)]

        # Only push the junction if it has future branches
        if len(remaining) > 0 and all(j[0] != vertex1 for j in activeJunction):
            activeJunction.append([vertex1, edge01])

    edge12, found_edge = vertex1.get_unhandled_edge(edge01)

    if not found_edge:
        if len(activeJunction) > 0:
            #print("hit an end!")
            # Backtrack through junctions until we find one with unhandled edges
            while len(activeJunction) > 0:
                vertex1, edge01 = activeJunction[-1]  # Most recent junction
                edge12, found_edge = vertex1.get_unhandled_edge(edge01)
                if found_edge:
                    vertex0 = edge01.get_other_vertex(vertex1)
                    break  # Found an edge, exit backtracking
                else:
                    activeJunction.pop()  # No edges left at this junction, remove it
            else:
                # If we exhausted all junctions
                #print("No edge found! Assuming tree is finished")
                break

        else:
            print("No edge found! Assuming tree search 1 is finished")
            break

    #print("Coordinates: ", vertex0.coords, vertex1.coords, found_edge)
    #PlotRodNetwork(vertexObjs, edgeObjs, [], [edge01, edge12])

    # Make sure every edge knows the edge it came from. This will be used to calculate twisting between beams
    edge12.root = edge01

    vertex2 = edge12.get_other_vertex(vertex1)
    
    # Parallel Transport frames from previous edge
    computeSpaceParallel_OO(edge01,edge12)

    # Compute material directors based on transported frames
    computeMaterialDirectors_OO(edge12)


    # Move on to next edge and vertex
    edge12.handled = True
    edge01 = edge12
    vertex0 = vertex1
    vertex1 = vertex2



# DO A SECOND TREE PASS. We need to do this to getKappa for each edge, since we don't yet know m1, m2 for the current edge in the first tree seach.
# Setup for second tree pass
for edge in edgeObjs:
    edge.handled = False
activeJunction = []
tree_search_active = True
edge01 = end_edge
vertex0 = first_end_vertex
print("Starting second Tree Pass")

while tree_search_active:
    vertex1 = edge01.get_other_vertex(vertex0)
    # Only add junction if it hasn't been added yet
    if vertex1.junction:
        # Get all outgoing edges except the one we came from
        remaining = [e for e in vertex1.edges if (not e.handled and e is not edge01)]

        # Only push the junction if it has future branches
        if len(remaining) > 0 and all(j[0] != vertex1 for j in activeJunction):
            activeJunction.append([vertex1, edge01])

    edge12, found_edge = vertex1.get_unhandled_edge(edge01)

    if not found_edge:
        if len(activeJunction) > 0:
            #print("hit an end!")
            # Backtrack through junctions until we find one with unhandled edges
            while len(activeJunction) > 0:
                vertex1, edge01 = activeJunction[-1]  # Most recent junction
                edge12, found_edge = vertex1.get_unhandled_edge(edge01)
                if found_edge:
                    vertex0 = edge01.get_other_vertex(vertex1)
                    break  # Found an edge, exit backtracking
                else:
                    activeJunction.pop()  # No edges left at this junction, remove it
            else:
                # If we exhausted all junctions
                #print("No edge found! Assuming tree is finished")
                break

        else:
            print("No edge found! Assuming tree is finished")
            break

    #PlotRodNetwork(vertexObjs, edgeObjs, [], [edge01, edge12])

    # Make sure every edge knows the edge it came from. This will be used to calculate twisting between beams
    edge12.root = edge01
    vertex2 = edge12.get_other_vertex(vertex1)

    # Compute Curvature and Twist
    getKappa_OO(vertex0, vertex1, vertex2, edge01, edge12)  # Natural curvature
    vertex1.rest_kappa = vertex1.kappa # Ensure rest kappa is set, and getKappa can be reused in the objective Function
    # Move on to next edge and vertex
    edge12.handled = True
    edge01 = edge12
    vertex0 = vertex1
    vertex1 = vertex2



# Set up boundary conditions: Index of fixed degrees of freedom. Form: [[VertexIndex, [X, Y, Z]]...]
vertexObjs[0].fix([True, True, True])
# Index of edges with fixed rotation. NOTE: Currently no edges have fixed rotation.
edgeObjs[0].twist_fixed = True



# Establish a Q_0 vector from degrees of freedom. It takes the for [x0, y0, z0, ... theta0, theta1, ...]
q_tr = np.zeros(len(vertexObjs)*3)
q_rot = np.zeros(len(edgeObjs))
for c, vertex in enumerate(vertexObjs):
    q_tr[c*3 : c*3+3] = np.array(vertex.coords)

for d, edge in enumerate(edgeObjs):
    q_rot[d] = edge.ref_twist
# TIME INTEGRATION LOOP
q_0 = np.concatenate([q_tr, q_rot])
print(q_0)

 # Assemble free_index vector
ndof_total = len(vertexObjs)*3 + len(edgeObjs) # Total number of DOFS:

free_index = list(range(ndof_total))
for vertex in vertexObjs:
    base_idx = vertex.index * 3
    for i, fixed in enumerate(vertex.dofs_fixed):
        if fixed:
            dof_idx = base_idx + i
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

for timeStep in range(Nsteps):


  q_new, u_new = objfun(end_edge,
                        qOld, uOld,
                        freeIndex, dt, tol,
                        massVector, massMatrix,
                        EA, EI, GJ, Fg)



  print('Current time: ', ctime, " Idx: ", timeStep)



  ctime += dt # Current time
  # Old parameters become new
  qOld = q_new.copy()
  uOld = u_new.copy()
  a1_old = a1_new.copy()
  a2_old = a2_new.copy()

