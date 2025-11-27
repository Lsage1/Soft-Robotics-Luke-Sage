import numpy as np
import matplotlib.pyplot as plt
from set_axes_equal import set_axes_equal


def plotShell(x0, ctime):

  x1 = x0[3:6]
  x2 = x0[6:9]
  x3 = x0[9:12]
  x0 = x0[0:3]

  fig = plt.figure(1)
  plt.clf()  # Clear the figure
  ax = fig.add_subplot(111, projection='3d')

  # Plot nodes
  X = np.array([x0[0], x1[0], x2[0], x0[0], x3[0], x1[0]])
  Y = np.array([x0[1], x1[1], x2[1], x0[1], x3[1], x1[1]])
  Z = np.array([x0[2], x1[2], x2[2], x0[2], x3[2], x1[2]])
  ax.plot3D(X, Y, Z, 'ko-')

  # Plot the first node with a red triangle
  ax.plot3D([X[0]], [Y[0]], [Z[0]], 'r^')

  # Set the title with current time
  ax.set_title(f't={ctime:.2f}')

  # Set axes labels
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')

  # Set equal scaling and a 3D view
  set_axes_equal(ax)
  plt.draw()  # Force a redraw of the figure

  plt.show()