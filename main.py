import math
from typing import Callable

import matplotlib.pyplot as plt


def explicit_3rd_order_taylor(a: float, h: float, y_0: float, n: int, func: Callable[[float, float], float]) -> list[
    float]:
    pass


def cauchy(a: float, h: float, y_0: float, n: int, func: Callable[[float, float], float]) -> list[float]:
    result = [y_0]

    for i in range(n):
        result.append(
            result[-1] + h * func((a + h * i) + h / 2, result[-1] + h / 2 * func(a + h * i, result[-1])))

    return result


def get_x_axis(a: float, h: float, n: int) -> list[float]:
    x_axis = [a]
    [x_axis.append(a + h * i) for i in range(n)]

    return x_axis


def get_result(a: float, y_0: float, method, exact: Callable[[float], float]):
    first_y = method(a, 0.1, y_0, 10, lambda x, y: 50 * y * (x - 0.6) * (x - 0.85))
    second_y = method(a, 0.05, y_0, 20, lambda x, y: 50 * y * (x - 0.6) * (x - 0.85))
    third_y = method(a, 0.025, y_0, 40, lambda x, y: 50 * y * (x - 0.6) * (x - 0.85))

    first_x = get_x_axis(a, 0.1, 10)
    second_x = get_x_axis(a, 0.05, 20)
    third_x = get_x_axis(a, 0.025, 40)

    exact_y = [exact(0)]
    [exact_y.append(exact(0 + 0.025 * i)) for i in range(40)]

    plt.plot(first_x, first_y, label="0.1", color="r")
    plt.plot(second_x, second_y, label="0.05", color="g")
    plt.plot(third_x, third_y, label="0.025", color="b")
    plt.plot(third_x, exact_y, label="Точное решение", color="black")

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(getattr(method, '__name__', 'Неизвестный метод'))

    plt.legend()
    plt.show()


get_result(0, 0.1, cauchy, lambda x: 1 / 10 * math.exp(1 / 12 * x * (200 * x * x - 435 * x + 306)))
