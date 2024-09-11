import numpy as np

# cost function
def f(x):
    return 3*x**2 + 2*x + 4 * np.sin(x)


def f_prime(x):
    return 6 * x + 2 + 4 * np.cos(x)

def gradient_descent(start_x, learning_rate, tol, max_epochs):
    x = start_x
    for i in range(max_epochs):
        grad = f_prime(x)
        if abs(grad) < tol:
            print(f"Hội tụ sau {i} vòng lặp.")
            return round(x, 6), i
        x = x - learning_rate * grad
    print(f"Không hội tụ sau {max_epochs} vòng lặp.") 
    return round(x, 6), max_epochs


start_x = -5
learning_rate = 0.1  
tolerance = 1e-3       
max_epochs = 10000     


minimum, iterations = gradient_descent(start_x, learning_rate, tolerance, max_epochs)
print("Giá trị x tại minimum:", minimum)
print("Giá trị nhỏ nhất của f(x):", round(f(minimum), 6))