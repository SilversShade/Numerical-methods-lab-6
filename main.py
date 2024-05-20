import math
import decimal
import numpy as np
from matplotlib import pyplot as plt
from typing import Callable


def f(x, y):
    return y + 3.6 + 0.9 * x * (x - 1) ** 2


def q(x):
    return 3.6 + 0.9 * x * (x - 1) ** 2


def get_analytical_solution(a: float, b: float, h: float, func: Callable[[float], float]) -> float:
    return zip(*[(x, func(x)) for x in np.arange(a, b + h, h)])


def runge_kutta_3rd_order(x0: float, mu: float, z0: float, h: float, n: int) -> list[(float, float, float)]:
    data = [(x0, mu, z0)]

    for i in range(1, n + 1):
        x_i = data[-1][0]
        y_i = data[-1][1]
        z_i = data[-1][2]

        x_next = x_i + h
        y_next = y_i + h * z_i

        k1 = h * f(x_i, y_i)
        k2 = h * f(x_i + h, y_i + k1)
        k3 = h * f(x_i + h / 2, y_i + (1 / 4) * k1 + (1 / 4) * k2)
        z_next = z_i + (1 / 6) * k1 + (1 / 6) * k2 + (4 / 6) * k3

        data.append((x_next, y_next, z_next))

    return data


def bisection_method(a: float, b: float, beta: float, x0: float, h: float, z0: float, n: int):
    result_left = runge_kutta_3rd_order(x0, a, z0, h, n)[-1][2]  # тут имеем z_n
    result_right = runge_kutta_3rd_order(x0, b, z0, h, n)[-1][2]  # тут имеем z_n

    while math.fabs(result_left - result_right) > 1e-9:
        mid = (a + b) / 2
        result_mid = runge_kutta_3rd_order(x0, mid, z0, h, n)[-1]  # тут имеем x_n, y_n, z_n
        if beta - result_mid[1] > result_mid[2]:
            a = mid
            result_left = result_mid[2]
        else:
            b = mid
            result_right = result_mid[2]

    return (a + b) / 2


def shooting_method(a: float, b: float, beta: float, x0: float, xn: float, h: float, z0: float):
    n = int((xn - x0) / h)
    mu = bisection_method(a, b, beta, x0, h, z0, n)
    result = runge_kutta_3rd_order(x0, mu, z0, h, n)
    return zip(*[(res[0], res[1]) for res in result])


def create_matrix(n: int, h: float, x0: float, xn: float):
    a0, b0, c0 = (0, -2 - h * h, 2)
    d0 = q(x0) * h * h - 12.6 * h

    A = [(a0, b0, c0)]
    D = [d0]

    for i in range(1, n):
        x_i = x0 + i * h
        a_i, b_i, c_i = (1 / (h * h), - (2 / (h * h) + 1), 1 / (h * h))
        d_i = q(x_i)

        A.append((a_i, b_i, c_i))
        D.append(d_i)

    a_n, b_n, c_n = (-2, h * h + 2 * h + 2, 0)
    d_n = (2 * math.exp(5) - 154.8) * 2 * h - q(xn) * h * h

    A.append((a_n, b_n, c_n))
    D.append(d_n)

    return A, D


def forward_run(A, D, n):
    lambda_set = [0]
    mu_set = [0]
    for i in range(n + 1):
        a_i, b_i, c_i = A[i]
        d_i = D[i]
        lambda_i = -c_i / (a_i * lambda_set[-1] + b_i)
        mu_i = (d_i - a_i * mu_set[-1]) / (a_i * lambda_set[-1] + b_i)
        lambda_set.append(lambda_i)
        mu_set.append(mu_i)

    return lambda_set, mu_set


def run(A, D, n):
    lambda_set, mu_set = forward_run(A, D, n)
    X = [mu_set[n + 1]] * (n + 1)
    for i in range(n, 0, -1):
        X[i - 1] = lambda_set[i] * X[i] + mu_set[i]
    return X


def finite_difference_method(h: float, x0: float, xn: float):
    n = int(5 / h)
    A, D = create_matrix(n, h, x0, xn)
    Y = run(A, D, n)
    X = [x for x in np.arange(0, 5 + step, step)]
    return X, Y


ox, oy = get_analytical_solution(0, 5, 0.05, lambda x: x * (-0.9 * x * x + 1.8 * x - 6.3) + math.exp(-x) + math.exp(x))
plt.plot(ox, oy, color="g", label="Точное решение")

while True:
    try:
        step = float(input("Введите шаг\n"))
        if step == 0:
            print('Шаг, равный нулю, недопустим.')
            continue
        if step < 0:
            print('Шаг не может быть отрицательным')
            continue
        if decimal.Decimal(str(5)) % decimal.Decimal(str(step)) != 0.0:
            print('Длина отрезка (5) должна нацело делиться на шаг.')
            continue
        break
    except:
        print('Введите корректное значение.')


ox, oy = finite_difference_method(step, 0, 5)
plt.plot(ox, oy, color="black", label="Метод прогонки")

ox, oy = shooting_method(1, 3, 2 * math.exp(5) - 154.8, 0, 5, step, -6.3)
plt.plot(ox, oy, color="r", label="Метод стрельбы")

plt.title(f"h = {step}")
plt.grid()
plt.legend()
plt.show()
