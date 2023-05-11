'''
This file contains the mathematics shown in the playlist of 5th module of Math for ML
Link : https://www.youtube.com/watch?v=VCF8kiLtBzU&list=PLfFghEzKVmjtZb9G6jvO9PLKvwUvK5avI&index=1

Here, four sections are explained :
1) Linear Algebra
2) Statistics
3) Probability
4) Calculus
'''

### Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### 1) Linear Algebra (5.1.X)
sns.set()

v1 = np.array([0, 0, 2, 3])
v2 = np.array([0, 0, 3, -2])
# vector addition
v3 = v1 - v2
v3 = 5 * v3
print(v3)

plt.quiver(0, 0, 2, 3, scale_units = 'xy', angles = 'xy', scale = 1, color = 'b')
plt.quiver(0, 0, 3, -2, scale_units = 'xy', angles = 'xy', scale = 1, color = 'y')
plt.quiver(0, 0, -5, 25, scale_units = 'xy', angles = 'xy', scale = 1, color = 'g')
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()

# Dot, Cross Product and Projection
# Representing as usual 1D array vectors (not as origin(x, y) and target(x, y))
a = np.array([3, 6, 1])
b = np.array([6, 10, 11])
a_dot_b = np.dot(a, b)
a_cross_b = np.cross(a, b)
print(f'a = {a}, b = {b}, a_dot_b = {a_dot_b}')
print(f'a = {a}, b = {b}, a_dot_b = {a_cross_b}')

# Projection of a on b = ((a_dot_b)/(magnitude_of_b ** 2)) * b
# The result is also a vector, hence the final multiplication with the vector b
magnitude_b = np.sqrt(sum(b ** 2))
a_proj_b = (a_dot_b / (magnitude_b ** 2)) * b
print(f'a = {a}, b = {b}, magnitude_b = {magnitude_b}, a_dot_b = {a_proj_b}')
