import numpy as np
from gradEb_hessEb import gradEb_hessEb

def getFb(q_all_new, qindex, m1, m2, kappaBar, kappa_junction, EI, voronoiRefLen, ne):
  # BENDING
  # Input q is a DOF vector of size 4*nv - 1
  ndof = len(q_all_new)

  Fb = np.zeros(ndof)
  Jb = np.zeros((ndof, ndof))

  node0 = q_all_new[qindex[0][1:4]]
  node1 = q_all_new[0:3]
  node2 = q_all_new[qindex[1][1:4]]
  m1e, m2e = m1[0][0], m2[0][0]
  m1f, m2f = m1[1][0], m2[1][0]
  curvature0 = kappa_junction[0]
  dL = 0.5 * (voronoiRefLen[0][0] + voronoiRefLen[1][0])
  dF, dJ = gradEb_hessEb(node0, node1, node2, m1e, m2e, m1f, m2f, curvature0, dL, EI)
  ind = np.concatenate([qindex[0][1:5],[0,1,2], qindex[1][0:4]])
  Fb[ind] -= dF
  Jb[np.ix_(ind, ind)] -= dJ

  node0 = q_all_new[qindex[1][1:4]]
  node1 = q_all_new[0:3]
  node2 = q_all_new[qindex[2][1:4]]
  m1e, m2e = m1[1][0], m2[1][0]
  m1f, m2f = m1[2][0], m2[2][0]
  curvature0 = kappa_junction[1]
  dL = 0.5 * (voronoiRefLen[1][0] + voronoiRefLen[2][0])
  dF, dJ = gradEb_hessEb(node0, node1, node2, m1e, m2e, m1f, m2f, curvature0, dL, EI)
  ind = np.concatenate([qindex[1][1:5], [0, 1, 2], qindex[2][0:4]])
  Fb[ind] -= dF
  Jb[np.ix_(ind, ind)] -= dJ

  node0 = q_all_new[qindex[2][1:4]]
  node1 = q_all_new[0:3]
  node2 = q_all_new[qindex[0][1:4]]
  m1e, m2e = m1[2][0], m2[2][0]
  m1f, m2f = m1[0][0], m2[0][0]
  curvature0 = kappa_junction[2]
  dL = 0.5 * (voronoiRefLen[2][0] + voronoiRefLen[0][0])
  dF, dJ = gradEb_hessEb(node0, node1, node2, m1e, m2e, m1f, m2f, curvature0, dL, EI)
  ind = np.concatenate([qindex[2][1:5], [0, 1, 2], qindex[0][0:4]])
  Fb[ind] -= dF
  Jb[np.ix_(ind, ind)] -= dJ

  for i in range(1,3):
      # # From the junction to the first node in each branch
      # node0 = q_all_new[0:3]
      # node1 = q_all_new[qindex[i][1 : 4]]
      # node2 = q_all_new[qindex[i][5 : 8]]
      #
      # m1e, m2e = m1[i][0], m2[i][0]
      # m1f, m2f = m1[i][1], m2[i][1]
      # dL = voronoiRefLen[i][1]
      # curvature0 = kappaBar[i][1]
      # dF, dJ = gradEb_hessEb(node0, node1, node2, m1e, m2e, m1f, m2f, curvature0, dL, EI)
      # ind = np.concatenate([[0, 1, 2], qindex[i][0 : 5], qindex[i][5 : 8]])
      # Fb[ind] -= dF
      # Jb[np.ix_(ind, ind)] -= dJ

      for c in range(ne-2): # Ignore the terminal nodes (0 and nv)

          node0 = q_all_new[qindex[i][4*c+1 : 4*c +4]] #
          node1 = q_all_new[qindex[i][4*c+5 : 4*c +8]]
          node2 = q_all_new[qindex[i][4*c+9 : 4*c +12]]
          m1e = m1[i][c+1]
          m2e = m2[i][c+1]
          m1f = m1[i][c+2]
          m2f = m2[i][c+2]

          dL = voronoiRefLen[i][c+2]
          curvature0 = kappaBar[i][c+2]

          dF, dJ = gradEb_hessEb(node0, node1, node2, m1e, m2e, m1f, m2f, curvature0, dL, EI)


          ind = np.concatenate([
              qindex[i][4 * c + 1: 4 * c +5],
              qindex[i][4 * c + 5: 4 * c + 9],
              qindex[i][4 * c + 9: 4 * c + 12]
          ])
          Fb[ind] -= dF
          Jb[np.ix_(ind, ind)] -= dJ

  return Fb, Jb