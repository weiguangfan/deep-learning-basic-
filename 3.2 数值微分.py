import numpy as np
import matplotlib.pyplot as plt

# def numerical_diff(f, x):
#     h = 10e-50
#     return (f(x + h) - f(x))/h


print(np.float32(1e-50))

def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h))/(2 * h)

def function_1(x):
    return 0.01 * x**2 + 0.1*x

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x,y)
plt.show()

print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))



