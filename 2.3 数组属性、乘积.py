import numpy as np
a = np.array([1, 2, 3, 4])
print(a)
print(np.ndim(a))
print(a.shape)
print(a.shape[0])

b = np.array([[1, 2], [3, 4], [5, 6]])
print(b)
print(b.shape)
print(np.ndim(b))

c = np.array([[1, 2], [3, 4]])
print(c.shape)
d = np.array([[5, 6], [7, 8]])
print(d.shape)
print(np.dot(c, d))

e = np.array([[1, 2, 3], [4, 5, 6]])
print(e.shape)
f = np.array([[1, 2], [3, 4], [5, 6]])
print(f.shape)
print(np.dot(e, f))

g = np.array([[1, 2], [3, 4], [5, 6]])
print(g.shape)
h = np.array([7, 8])
print(h.shape)
print(np.dot(g, h))


