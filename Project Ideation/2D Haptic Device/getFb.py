import numpy as np
from gradEb import gradEb
from hessEb import hessEb


import numpy as np
from gradEb import gradEb
from hessEb import hessEb

def getFb(q, EI, deltaL, vertices):
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
  for k in range(1, N-1):
    xkm1 = q[2*k-2] # x coordinate of the first node
    ykm1 = q[2*k-1] # y coordinate of the first node
    xk = q[2*k] # x coordinate of the second node
    yk = q[2*k+1] # y coordinate of the second node
    xkp1 = q[2*k+2] # x coordinate of the third node
    ykp1 = q[2*k+3] # y coordinate of the third node
    ind = np.arange(2*k-2, 2*k+4)

    # Rest Geometries from vertices:
    vkm1 = vertices[k-1]
    vk = vertices[k]
    vkp1 = vertices[k+1]

    # Get rest edges
    e1_rest = vkm1 - vk
    e2_rest = vkp1 - vk

    # Get Rest angle
    dotprod = np.dot(e1_rest, e2_rest)
    n1 = np.linalg.norm(e1_rest)
    n2 = np.linalg.norm(e2_rest)
    #cos_theta = np.clip(dotprod / (n1*n2), -1, 1) # Make sure nothing out of range -1, 1
    rest_angle = np.atan2(np.cross(e1_rest, e2_rest), dotprod)

    # Get rest voronoi length
    lkm1 = np.linalg.norm(vk-vkm1)
    lk = np.linalg.norm(vkp1-vk)
    rest_deltaL = 0.5 * (lkm1 + lk)

    gradEnergy = gradEb(xkm1, ykm1, xk, yk, xkp1, ykp1, 0, rest_deltaL, EI)
    hessEnergy = hessEb(xkm1, ykm1, xk, yk, xkp1, ykp1, 0, rest_deltaL, EI)

    Fb[ind] -= gradEnergy # force = - gradient of energy. Fb is the stretching force
    Jb[np.ix_(ind, ind)] -= hessEnergy # index vector: 0:6

  return Fb, Jb