import numpy as np

# Declaring the initial distribution
v = np.array([[1.0, 0.0]])
# Declaring the Transition Matrix T
T = np.array([[0.90, 0.10],
              [0.50, 0.50]])

# Obtaining T after 3 steps
T_3 = np.linalg.matrix_power(T, 3)
# Obtaining T after 50 steps
T_50 = np.linalg.matrix_power(T, 50)
# Obtaining T after 100 steps
T_100 = np.linalg.matrix_power(T, 100)

# Printing the initial distribution
print("v: " + str(v))
print("v_1: " + str(np.dot(v, T)))
print("v_3: " + str(np.dot(v, T_3)))
print("v_50: " + str(np.dot(v, T_50)))
print("v_100: " + str(np.dot(v, T_100)))
