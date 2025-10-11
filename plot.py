from matplotlib import pyplot as plt


def plot(x, index_matrix, t):
  plt.figure() # Create a new figure for each plot
  plt.title(f'Time: {t:.2f} second') # Add a title with the current time
  for i in range(index_matrix.shape[0]):
    ind = index_matrix[i].astype(int)  # Convert indices to integers
    xi = x[ind[0]]
    yi = x[ind[1]]
    xj = x[ind[2]]
    yj = x[ind[3]]
    plt.plot([xi, xj], [yi, yj], 'bo-') # Plot a blue line with circles at the nodes
  plt.xlabel('x')
  plt.ylabel('y')
  plt.axis('equal') # Ensure equal scaling for x and y axes
  plt.grid(True)
  plt.show()