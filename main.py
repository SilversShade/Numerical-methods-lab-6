import math
from typing import Callable

import matplotlib.pyplot as plt


class Method:
    def __init__(self, title: str, func: Callable):
        self.title = title
        self.func = func


def runge_kutta_3rd_order_precalc(x_0: float, y_0: float, h: float, func: Callable[[float, float], float]) -> float:
    k1 = h * func(x_0, y_0)
    k2 = h * func(x_0 + h, y_0 + k1)
    k3 = h * func(x_0 + h / 2, y_0 + (1 / 4) * k1 + (1 / 4) * k2)

    return y_0 + (1 / 6) * k1 + (1 / 6) * k2 + (4 / 6) * k3


def implicit_two_step_adams(a: float, h: float, y_0: float, n: int, func: Callable[[float, float], float]) -> list[
    float]:
    result = [y_0, runge_kutta_3rd_order_precalc(0, y_0, h, func)]

    for i in range(1, n):
        y_n = result[-1]
        x_n = a + h * i

        result.append((y_n + (8 / 12) * h * func(x_n, y_n) - (1 / 12) * h * func(a + h * (i - 1), result[-2]))
                      /
                      (1 - (5 / 12) * h * 50 * ((a + h * (i + 1)) - 0.6) * ((a + h * (i + 1)) - 0.85)))

    return result


def explicit_3rd_order_taylor(a: float, h: float, y_0: float, n: int, func: Callable[[float, float], float]) -> list[
    float]:
    result = [y_0]

    for i in range(n):
        x_i = a + h * i
        y_i = result[-1]
        result.append(y_i + h * func(x_i, y_i)
                      + h * h / 2 * (
                              100 * x_i * y_i - 72.5 * y_i + func(x_i, y_i) * (50 * x_i * x_i - 72.5 * x_i + 25.5))
                      + h * h * h / 6 * (100 * y_i + func(x_i, y_i) * (200 * x_i - 145) - 100 * x_i * y_i
                                         - 72.5 * y_i + func(x_i, y_i) * (50 * x_i * x_i - 72.5 * x_i + 25.5)
                                         * (-50 * x_i * x_i + 72.5 * x_i - 25.5)))
    return result


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


def get_result(a: float, y_0: float, methods: list[Method], exact: Callable[[float], float]):
    figure, axis = plt.subplots(2, 2)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    for idx, pos in enumerate([[0, 0], [0, 1], [1, 0]]):
        first_y = methods[idx].func(a, 0.1, y_0, 10, lambda x, y: 50 * y * (x - 0.6) * (x - 0.85))
        second_y = methods[idx].func(a, 0.05, y_0, 20, lambda x, y: 50 * y * (x - 0.6) * (x - 0.85))
        third_y = methods[idx].func(a, 0.025, y_0, 40, lambda x, y: 50 * y * (x - 0.6) * (x - 0.85))

        first_x = get_x_axis(a, 0.1, 10)
        second_x = get_x_axis(a, 0.05, 20)
        third_x = get_x_axis(a, 0.025, 40)

        exact_y = [exact(0)]
        [exact_y.append(exact(0 + 0.025 * i)) for i in range(40)]

        axis[pos[0], pos[1]].plot(first_x, first_y, color="r")
        axis[pos[0], pos[1]].plot(second_x, second_y, color="g")
        axis[pos[0], pos[1]].plot(third_x, third_y, color="b")
        axis[pos[0], pos[1]].plot(third_x, exact_y, color="black")

        axis[pos[0], pos[1]].title.set_text(methods[idx].title)

    axis[1, 1].axis('off')
    figure.legend(['0.1', '0.05', '0.025', 'Точное решение'], loc='lower right')

    plt.show()


get_result(0, 0.1, [
    Method('Коши', cauchy),
    Method('Тейлор явный 3 порядка', explicit_3rd_order_taylor),
    Method('Адамса двушаговый неявный', implicit_two_step_adams)
], lambda x: 1 / 10 * math.exp(1 / 12 * x * (200 * x * x - 435 * x + 306)))
