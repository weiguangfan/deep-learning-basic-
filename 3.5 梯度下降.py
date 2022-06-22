import numpy as np



def function_2(x):
    print("x: ", x)
    return x[0]**2 + x[1]**2

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    print("grad: ", grad)
    for idx in range(x.size):
        print("idx: ", idx)
        tmp_val = x[idx]
        print('i:tmp_val: ', idx, tmp_val)
        x[idx] = tmp_val + h
        fxh1 = f(x)
        print("idx:fxh1: ", idx, fxh1)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2)/(2*h)
        print("grad: ", grad)
        x[idx] = tmp_val
    return grad

def gradient_descent(f, init_x, lr = 0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        print("###" * 10)
        print("i: ", i)
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

init_x = np.array([-3.0, 4.0])
# gradient_descent(function_2, init_x, lr=0.1, step_num=100)
# gradient_descent(function_2, init_x, lr=10, step_num=100)
# gradient_descent(function_2, init_x, lr=1e-10, step_num=100)




