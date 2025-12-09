import numpy as np
from gradEs_hessEs import gradEs_hessEs

def getFS_Junction(q, EA, refLen, refLenJunction):
  # STRETCHING
  # Input q is a DOF vector of size 4*nv - 1
  # Output is the stretching force vector (size 4*nv-1) and it's gradient w.r.t. q (square matrix)
  print("inGetFS Junction")
  numRods = len(q)

  q_all = np.concatenate(q)

  q_junction = q[0][:3]
  q[0] = q[0][3:]  # Remove the first 3 elements of q[1] because they will be shared with q[0] at the junction
  q[1] = q[1][3:]
  q[2] = q[2][3:]
  index = []
  jun_index = np.arange(0,3)
  index.append(np.arange(len(jun_index), len(q[0])+len(jun_index))) # Get the indices of the first vector
  index.append(np.arange(len(jun_index)+len(index[0]), len(q[0])+len(q[1])+len(jun_index)))
  index.append(np.arange(len(jun_index)+len(index[0])+len(index[1]), len(index[0])+len(index[1]) + len(q[2])+len(jun_index)))

  print(jun_index, index)




  ndof = len(q_all)

  ne = int(len(q[0])/4) # Get the number of edges per branch
  print(ne)

  Fs = np.zeros(ndof)
  Js = np.zeros((ndof, ndof))


  for i in range(numRods):
    ## -- TAKE CARE OF JUNCTION FIRST --
    xa = q_all[:3]
    xb = q[i][1:4]
    ind = np.array([0, 1, 2, index[i][1], index[i][2], index[i][3]])
    print("Xa,xb", xa, xb, ind, refLen[i][0])
    dL = refLen[i][0]
    dF, dJ = gradEs_hessEs(xa, xb, dL, EA)
    Fs[ind] -= dF
    Js[np.ix_(ind, ind)] -= dJ

    ## NOW MOVE ON TO the rest of the rods! ##
    for c in range(ne-1): # Subract 1 edge because we handled it at the junction
      xa = q[i][4*c+1:4*c+4]
      xb = q[i][4*c+5:4*c+8]
      ind = np.array([index[i][c*4+1], index[i][c*4+2], index[i][c*4+3], index[i][c*4+5], index[i][c*4+6], index[i][c*4+7]])
      print("Xa,xb", xa, xb, ind, refLen[i][c+1])
      dL = refLen[i][c+1]
      dF, dJ = gradEs_hessEs(xa, xb, dL, EA)
      Fs[ind] -= dF
      Js[np.ix_(ind, ind)] -= dJ

  print(Fs, Js)

  return Fs, Js