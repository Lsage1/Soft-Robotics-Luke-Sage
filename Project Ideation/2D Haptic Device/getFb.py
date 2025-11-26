import numpy as np
from gradEb import gradEb
from hessEb import hessEb


import numpy as np
from gradEb import gradEb
from hessEb import hessEb

def getFb(q, EI, vertexObjs, edgeObjs):
  # q - DOF vector of size N
  # EI - bending stiffness
  # deltaL - undeformed Voronoi length (assume to be a scalar for this simple example)
  # Output:
  # Fb - a vector (negative gradient of elastic stretching force)
  # Jb - a matrix (negative hessian of elastic stretching force)

  ndof = q.size # Number of DOFs
  N = ndof // 2 # Number of nodes

  Fb = np.zeros(ndof) # bending force
  Jb = np.zeros((ndof, ndof))

  # First bending spring (USE A LOOP for the general case)
  for k, vertex in enumerate(vertexObjs):
    for e, edgePair in enumerate(vertex.edgePairs):
        prev_vertex = edgePair[0].get_other_vertex(vertex)
        next_vertex = edgePair[1].get_other_vertex(vertex)

        # Get current, next, and previous index based on vertex
        cind = vertex.index
        pind = prev_vertex.index
        nind = next_vertex.index

        xkm1 = q[2*pind] # x coordinate of the first node
        ykm1 = q[2*pind+1] # y coordinate of the first node
        xk = q[2*cind] # x coordinate of the second node
        yk = q[2*cind+1] # y coordinate of the second node
        xkp1 = q[2*nind] # x coordinate of the third node
        ykp1 = q[2*nind+1] # y coordinate of the third node

        ind = np.array([2*pind, 2*pind+1, 2*cind, 2*cind+1, 2*nind, 2*nind+1])

        # Calculate rest voronoi lengths
        rest_voronoi_L = edgePair[0].rest_length + edgePair[1].rest_length


        # Bending force scaled between edge pairs reduced if a junction exists
        junctionFactor = 1 / len(vertex.edgePairs)

        # edge pair rest bending angle (index of edge pair is the same as index of rest angles stored)
        curvature0 = vertex.rest_angles[e]

        gradEnergy = gradEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, rest_voronoi_L, EI)
        hessEnergy = hessEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, rest_voronoi_L, EI)

        Fb[ind] -= gradEnergy * junctionFactor # force = - gradient of energy. Fb is the stretching force
        Jb[np.ix_(ind, ind)] -= hessEnergy * junctionFactor# index vector: 0:6

  return Fb, Jb