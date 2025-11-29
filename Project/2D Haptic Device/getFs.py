import numpy as np
from gradEs import gradEs
from hessEs import hessEs



def getFs(q, EA, edgeObjs):
  # q - DOF vector of size N
  # EA - stretching stiffness
  # deltaL - undeformed reference length (assume to be a scalar for this simple example)
  # Output:
  # Fs - a vector (negative gradient of elastic stretching force)
  # Js - a matrix (negative hessian of elastic stretching force)

  ndof = q.size # Number of DOFs
  N = ndof // 2 # Number of nodes

  Fs = np.zeros(ndof) # stretching force
  Js = np.zeros((ndof, ndof))

  for k, edge in enumerate(edgeObjs):
      v1_index = edge.vertex1.index
      v2_index = edge.vertex2.index
      # k-th stretching spring (USE A LOOP for the general case
      xkm1 = q[2*v1_index] # x coordinate of the first node
      ykm1 = q[2*v1_index+1] # y coordinate of the first node
      xk = q[2*v2_index] # x coordinate of the second node
      yk = q[2*v2_index+1] # y coordinate of the second node

      rest_length = edge.rest_length

      ind = np.array([2*v1_index, 2*v1_index+1, 2*v2_index, 2*v2_index+1]) # 0, 1, 2, 3 for k = 0
      gradEnergy = gradEs(xkm1, ykm1, xk, yk, rest_length, EA)
      hessEnergy = hessEs(xkm1, ykm1, xk, yk, rest_length, EA)

      Fs[ind] -= gradEnergy # force = - gradient of energy. Fs is the stretching force
      Js[np.ix_(ind, ind)] -= hessEnergy # index vector: 0:4

  return Fs, Js