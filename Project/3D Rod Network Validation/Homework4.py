import subprocess

#Task 1: Given a characteristic force, find a steady state value.
subprocess.run(["python", "3D Haptic Node Simple.py"])

# Task 2: Sweep over a large number of forces to determine the spring stiffness of the helix
subprocess.run(["python", "Task2.py"])

# Task 3: For a number of helix diameters, determine spring stiffness.
subprocess.run(["python", "Task3.py"])