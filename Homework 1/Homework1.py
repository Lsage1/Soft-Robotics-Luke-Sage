import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

from getFexternal import getFexternal
from myInt import myInt
from myIntExp import myIntExp
from plot import plot

# Course Provided: Input and interpret Nodes

nodes_file_path = 'nodes.txt'
node_coordinates = []

try:
    with open(nodes_file_path, 'r') as f:
        for line in f:
            # Split each line by comma and remove leading/trailing whitespace
            parts = [part.strip() for part in line.split(',')]
            # Assuming the format is node number, x, y
            # We only need x and y, which are the second and third elements (index 1 and 2)
            if len(parts) == 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    node_coordinates.append([x, y])
                except ValueError:
                    print(f"Skipping line due to non-numeric coordinates: {line.strip()}")
            else:
                print(f"Skipping line due to incorrect format: {line.strip()}")

    # Convert the list of coordinates to a NumPy array

    node_matrix = np.array(node_coordinates)

    print("Node coordinates successfully loaded into a numpy matrix.")
    display(node_matrix)

except FileNotFoundError:
    print(f"Error: The file '{nodes_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

free_nodes_file_path = 'free_nodes.txt'

try:
    with open(free_nodes_file_path, 'r') as f:
        line = f.readline()
        free_nodes = [part.strip() for part in line.split(',')]
        print("free nodes: ", free_nodes)
except FileNotFoundError:
    print(f"Error: The file '{free_nodes_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Course Provided: Import Springs

springs_file_path = 'springs.txt'
index_info = []
stiffness_info = []
try:
    with open(springs_file_path, 'r') as f:
        for line in f:
            # Split each line by comma and remove leading/trailing whitespace
            parts = [part.strip() for part in line.split(',')]
            # Assuming the format is spring number, first node, second node, stiffness
            if len(parts) == 3:
                try:
                    first_node_index = float(parts[0])
                    second_node_index = float(parts[1])
                    stiffness = float(parts[2])
                    index_info.append([2 * first_node_index, 2 * first_node_index + 1, 2 * second_node_index,
                                       2 * second_node_index + 1])
                    stiffness_info.append(stiffness)
                except ValueError:
                    print(f"Skipping line due to non-numeric coordinates: {line.strip()}")
            else:
                print(f"Skipping line due to incorrect format: {line.strip()}")

    # Convert the list of coordinates to a NumPy array
    index_matrix = np.array(index_info)
    stiffness_matrix = np.array(stiffness_info)

    print("Spring indices successfully loaded into a numpy matrix.")
    display(index_matrix)

    print("Spring stiffnesses successfully loaded into a numpy matrix.")
    display(stiffness_matrix)
except FileNotFoundError:
    print(f"Error: The file '{springs_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")


# Preparation @ T=0

N = node_matrix.shape[0] # Number of nodes
ndof = 2 * N # Number of degrees of freedom

# Initialize positions, velocities, and accelerations
x_old = np.zeros(ndof)
u_old = np.zeros(ndof)
a_old = np.zeros(ndof)

# Build the position (DOF) vector. Velocity and acceleration remains zero
for i in range(N):
  x_old[2*i] = node_matrix[i][0]
  x_old[2*i+1] = node_matrix[i][1]

# Every spring has a rest length
l_k = np.zeros_like(stiffness_matrix)
for i in range(stiffness_matrix.shape[0]):
  ind = index_matrix[i].astype(int)  # Convert indices to integers
  xi = x_old[ind[0]]
  yi = x_old[ind[1]]
  xj = x_old[ind[2]]
  yj = x_old[ind[3]]
  l_k[i] = np.sqrt((xj - xi)**2 + (yj - yi)**2)

# Setup mass and weight
# Mass
m = np.zeros(ndof)
for i in range(ndof):
  m[i] = 1.0

# Weight
W = getFexternal(m)

#############################################################################################
# Main Simulation Loop:

dt = 0.01 # Time step size
maxTime = 1  # total time of simulation
t = np.arange(0, maxTime + dt, dt)

# free indices
#free_DOF = np.arange(2, ndof - 2)

free_DOF = []
for i in free_nodes:
    i = int(i)
    free_DOF.append(i*2)
    free_DOF.append(i*2+1)
#print(ndof, free_DOF)

# Container to store y-coordinate of middle node
y_middle = np.zeros(len(t))
y_middle[0] = x_old[3] # y-coordinate of middle node
y_middle_s = np.zeros(len(t))
y_middle_s[0] = x_old[7] # y-coordinate of middle node


plot(x_old, index_matrix, 0)

for k in range(len(t)-1):
  t_new = t[k+1]
  #print(k, round(t_new, 2))


  #x_new, u_new = myIntExp(t_new, x_old, u_old, free_DOF, stiffness_matrix, index_matrix, m, dt, l_k) # I added l_k to this function
  x_new, u_new = myInt(t_new, x_old, u_old, free_DOF, stiffness_matrix, index_matrix, m, dt, l_k) # I added l_k to this function

  #if k % 10 == 0:
  if t_new in [0, 0.1, 1, 10, 100]:
    print("output a graph at t =", t_new)
    plot(x_new, index_matrix, t_new)
  y_middle[k+1] = x_new[3]
  y_middle_s[k + 1] = x_new[7]

  x_old = x_new
  u_old = u_new

# Plot
plt.figure()
plt.plot(t, y_middle, 'ro-')
plt.plot(t, y_middle_s, 'bo-')
plt.xlabel('Time (s)')
plt.ylabel('Middle Node y-coordinate')
plt.title('Middle Node y-coordinate vs. Time')
plt.grid(True)
plt.show()

