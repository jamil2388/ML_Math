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
# plt.show()

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

# Matrix
# Create a 2*3 Matrix
matrix1 = np.array([[2, 3, -1], [11, 7, 9]])
matrix2 = np.array([[2, 1, 1], [1, -5, 4], [6, -9, 2]])
matrix3 = np.ones((5, 2), dtype = int)
matrix4 = np.zeros((5, 5), dtype = int)
# Notice how np.eye and np.random.rand take in the size as separate numbers instead of tuples
matrix5 = np.eye(5, 5, dtype=  int)
# the shape property is actually a tuple object, so it is not a callable function like 'shape()'
print(f'matrix1 = \n{matrix1}\nShape = {matrix1.shape}')
print(f'matrix2 = \n{matrix2}\n')
print(f'matrix3 = \n{matrix3}\n')
print(f'matrix4 = \n{matrix4}\n')
print(f'matrix5 = \n{matrix5}\n')

# create matrix with random values
random_matrix = np.random.rand(3, 4)
# the example from manual shows that the 100 here is the 'high' and the low is taken to be 0
random_int_matrix = np.random.randint(100, size = (3, 4))
print(f'random_matrix = \n{random_matrix}\n')
print(f'random_int_matrix = \n{random_int_matrix}\n')

# Transpose
random_int_matrix_t = np.transpose(random_int_matrix)
print(f'random_int_matrix_t = \n{random_int_matrix_t}\n')

