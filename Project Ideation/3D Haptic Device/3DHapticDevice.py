import numpy as np
import matplotlib.pyplot as plt
from plotrod_simple import plotrod_simple
from computeTangent import computeTangent
from computeSpaceParallel import computeSpaceParallel
from computeMaterialDirectors import computeMaterialDirectors
from getKappa import getKappa
from objfun import objfun
from computeKappa import computeKappa
from PlotRodNetwork import PlotRodNetwork
from GeometryObjects import EdgeObj
from GeometryObjects import VertexObj

print("Running Task 1")

vertices = np.array([[0,0,0], [0,0.05,0], [0.05,0,0], [0,-0.05,0], [-0.05, 0,0], [0,0.1,-0.05], [0.1,0,-0.05], [0,-0.1,-0.05], [-0.1, 0,-0.05]])
edges = np.array([[0,1], [0,2], [0,3], [0,4], [1,5], [2,6], [3,7], [4,8]])

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
    edgeObj = EdgeObj(v1, v2) # Create a new edge object with the two vertices that define it
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

    if len(vertex.edgePairs) > 1:
        vertex.junction = True
    if len(vertex.edgePairs) == 0:
        vertex.end = True



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



# Helix parameters
r0 = 0.001 # cross-sectional radius of the rod # Given, d = 0.002 m
D = 0.04 # meter: helix diameter
pitch = 2 * r0 # Pitch is the same as the cross-sectional diameter
N = 5 # Number of turns
# a and b are parameters used in standard (wikipedia) definition of helix
a = D/2 # Helix radius
b = pitch / (2.0 * np.pi)
T = 2.0 * np.pi * N # Angle created by the helix (N turns in the center)
L = T * np.sqrt( a**2 + b ** 2) # Arc length of the helix
axial_l = N * pitch # Axial length


Estimated_Arc = np.pi * D * N


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
totalM = L * np.pi * r0**2 * rho  # Total mass of the rod
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


# COMPUTE THE FRAMES

# Reference frame (At t=0, we initialize it with space parallel reference frame but not mandatory)
tangent = computeTangent(qOld_tr)

# NOTE:  For each "Start point" do this. Calculate a reference vector, considered the origin of no rotation
print(len(qOld_tr), "Translational Degrees of Freedom")
print(len(qOld_rot), "Rotational Degrees of Freedom")
for vertex in vertexObjs:
    if vertex.end == True:
        t0 = tangent[vertex.index, :]
        print(t0)
        # Maybe save the q0
        # At the end node, create a reference vector.
        arb_v = np.array([0, 0, -1])
        a1_first = np.cross(t0, arb_v) / np.linalg.norm(np.cross(t0, arb_v))

        if np.linalg.norm(np.cross(t0, arb_v)) < 1e-3: # Check if t0 and arb_v are parallel
          arb_v = np.array([0, 1, 0])
          a1_first = np.cross(t0, arb_v) / np.linalg.norm(np.cross(t0, arb_v))

        # NOTE: Need to generate a vector of q_segment that follows each END to a JUNCTION or END
        #getQSegment(vertex, vertexObjs, edgeObjs)

        # q_segment will need to be passed. computeSpaceParallel will need to be changed not have 4 inputs
        #a1, a2 = computeSpaceParallel(a1_first, q_segment)

quit()

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

  # Assemble a list of times to check against the current time. Used to check if steady state is reached.
  if ctime > track_time:
      # assemble a list of the times to check against
      track_list = endZ[timeStep-(track_steps-1) : timeStep+1]

      # Obtain a list of the percent differences of each value within the check time
      for j, i in enumerate(track_list[1:]):
          zdiff_list[j] = ( (i - track_list[0]) / track_list[0] )
      # Check if all the values within the check time are within 1% of the current time
      satisfied = True
      for z in zdiff_list:
          if abs(z) > 0.01:
              satisfied = False
      if satisfied:
          breakStep = timeStep
          break

  print('Current time: ', ctime, " Idx: ", timeStep, " Satisfied: ", satisfied)




  if timeStep % 10 == 0:
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
plt.plot(time_array[:breakStep], endZ[:breakStep], 'ro-')
plt.plot(time_array[breakStep], endZ[breakStep], 'bo', label = "Steady State Value")

plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('End Z (m)')
plt.show()

print("Finished Task 1")
