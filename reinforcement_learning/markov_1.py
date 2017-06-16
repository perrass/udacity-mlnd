import numpy as np


T = np.array([[0.90, 0.10], [0.50, 0.50]])

T_3 = np.linalg.matrix_power(T, 3)

T_50 = np.linalg.matrix_power(T, 50)

T_100 = np.linalg.matrix_power(T, 100)

print("T: " + str(T))
print("T_3: " + str(T_3))
print("T_50: " + str(T_50))
print("T_100: " + str(T_100))
