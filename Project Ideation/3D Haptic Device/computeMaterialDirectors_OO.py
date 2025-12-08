import numpy as np

def computeMaterialDirectors_OO(edge):
  #a1 = First reference director
  #a2 = second reference director
    a1 = edge.a1
    a2 = edge.a2

    cs = np.cos(edge.theta)
    sn = np.sin(edge.theta)
    edge.m1 = cs * a1 + sn * a2
    edge.m2 = - sn * a1 + cs * a2

    return