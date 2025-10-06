import sympy as sp


def generate_butcher_tableau(x_order, y_order):
    if y_order <= x_order:
        raise ValueError("y_order must be greater than x_order for an embedded method.")

    s = x_order + 1  # Number of stages (heuristic choice)
    A = sp.Matrix(s, s, lambda i, j: sp.Symbol(f'a{i + 1}{j + 1}') if j < i else 0)
    c = sp.Matrix(s, 1, lambda i, _: sp.Symbol(f'c{i + 1}'))
    b = sp.Matrix(1, s, lambda _, j: sp.Symbol(f'b{j + 1}'))
    b_hat = sp.Matrix(1, s, lambda _, j: sp.Symbol(f'b_hat{j + 1}'))

    tableau = {
        'A': A,
        'c': c,
        'b': b,
        'b_hat': b_hat
    }

    return tableau


# Example usage
x = 8
y = 13
bt = generate_butcher_tableau(x, y)
for key, value in bt.items():
    print(f"{key}:\n{value}\n")
