import numpy as np
import matplotlib.pyplot as plt
from set_axes_equal import set_axes_equal


def plotShell(q, edges, ctime):

    nv = q.size // 3
    pts = q.reshape((nv, 3))

    fig = plt.figure(1)
    plt.clf()  # Clear the figure
    ax = fig.add_subplot(111, projection='3d')

    for (i,j) in edges:
        xi, yi, zi = pts[i]
        xj, yj, zj = pts[j]
        ax.plot([xi, xj], [yi, yj], [zi, zj], color='r')

    for k in range(nv):
        x,y,z = pts[k]
        ax.plot([x], [y], [z], 'ko')


    # Plot the first node with a red triangle
    # ax.plot3D([X[0]], [Y[0]], [Z[0]], 'r^')

    # Set the title with current time
    ax.set_title(f't={ctime:.2f}')

    # Set axes labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Set equal scaling and a 3D view
    set_axes_equal(ax)
    plt.draw()  # Force a redraw of the figure

    #plt.show()