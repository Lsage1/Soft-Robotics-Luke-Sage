import numpy as np
from gradEs_hessEs import gradEs_hessEs

def getFS_Junction(q_all_new, EA, refLen, jun_index, index, numRods, nv):
  # STRETCHING
  # Input q is a DOF vector of size 4*nv - 1
  # Output is the stretching force vector (size 4*nv-1) and it's gradient w.r.t. q (square matrix)
  print("inGetFS Junction")


  ndof = len(q_all_new)

  ne = nv-1 # Get the number of edges per branch

  Fs = np.zeros(ndof)
  Js = np.zeros((ndof, ndof))

  print(q_all_new)

  for i in range(numRods):
    ## -- TAKE CARE OF JUNCTION FIRST --
    xa = q_all_new[:3]
    xb = q_all_new[index[i]][1:4]
    ind = np.array([0, 1, 2, index[i][1], index[i][2], index[i][3]])
    print("Xa,xb", xa, xb, ind, refLen[i][0])
    dL = refLen[i][0]
    dF, dJ = gradEs_hessEs(xa, xb, dL, EA)
    Fs[ind] -= dF
    Js[np.ix_(ind, ind)] -= dJ

    ## NOW MOVE ON TO the rest of the rods! ##
    for c in range(ne-1): # Subract 1 edge because we handled it at the junction
      xa = q_all_new[index[i]][4*c+1:4*c+4]
      xb = q_all_new[index[i]][4*c+5:4*c+8]
      ind = np.array([index[i][c*4+1], index[i][c*4+2], index[i][c*4+3], index[i][c*4+5], index[i][c*4+6], index[i][c*4+7]])
      print("Xa,xb", xa, xb, ind, refLen[i][c+1])
      dL = refLen[i][c+1]
      dF, dJ = gradEs_hessEs(xa, xb, dL, EA)
      Fs[ind] -= dF
      Js[np.ix_(ind, ind)] -= dJ


  return Fs, Js