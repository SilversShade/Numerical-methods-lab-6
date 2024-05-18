import math
import numpy as np
from matplotlib import pyplot as plt
from typing import Callable


def f(x, y):
    return y + 2.8 + 0.7 * x * (x - 1) ** 2


def get_analytical_solution(a: float, b: float, h: float, func: Callable[[float], float]) -> float:
    return zip(*[(x, func(x)) for x in np.arange(a, b + h, h)])


def cauchy(x0: float, y0: float, z0: float, h: float) -> (float, float):
    x_next = x0 + h
    y_next = y0 + h * z0
    z_next = z0 + h * f(x0 + (h / 2), y0 + (h / 2) * f(x0, y0))

    return x_next, y_next, z_next


def explicit_2nd_order_adams(x0: float, mu: float, z0: float, h: float, n: int) -> list[(float, float, float)]:
    data = [(x0, mu, z0), cauchy(x0, mu, z0, h)]

    for i in range(2, n + 1):
        x_i = data[-1][0]
        y_i = data[-1][1]
        z_i = data[-1][2]

        x_next = x_i + h
        y_next = y_i + (h / 2) * (3 * z_i - data[-2][2])
        z_next = z_i + (h / 2) * (3 * f(x_i, y_i) - f(data[-2][0], data[-2][1]))

        data.append((x_next, y_next, z_next))

    return data


def bisection_method(a: float, b: float, beta: float, x0: float, h: float, z0: float, n: int):
    result_left = explicit_2nd_order_adams(x0, a, z0, h, n)[-1][2]  # тут имеем z_n
    result_right = explicit_2nd_order_adams(x0, b, z0, h, n)[-1][2]  # тут имеем z_n

    while math.fabs(result_left - result_right) > 1e-9:
        mid = (a + b) / 2
        result_mid = explicit_2nd_order_adams(x0, mid, z0, h, n)[-1][2]  # тут имеем z_n
        if beta > result_mid:
            a = mid
            result_left = result_mid
        else:
            b = mid
            result_right = result_mid

    return (a + b) / 2


def shooting_method(a: float, b: float, beta: float, x0: float, xn: float, h: float, z0: float):
    n = int((xn - x0) / h)
    mu = bisection_method(a, b, beta, x0, h, z0, n)
    result = explicit_2nd_order_adams(x0, mu, z0, h, n)

    return zip(*[(res[0], res[1]) for res in result])


ox, oy = get_analytical_solution(0, 5, 0.05, lambda x: math.exp(x) + math.exp(-x) + x * (-0.7 * x * x + 1.4 * x - 4.9))
plt.plot(ox, oy, color="b", label="Точное решение")

step = float(input("Введите шаг\n"))
ox, oy = shooting_method(1, 3, math.exp(5) - math.exp(-5) - 43.4, 0, 5, step, -4.9)
plt.plot(ox, oy, color="r", label="Метод стрельбы")

plt.title(f"Шаг h = {step}")
plt.grid()
plt.legend()
plt.show()
