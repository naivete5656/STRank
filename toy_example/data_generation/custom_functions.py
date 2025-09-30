import numpy as np


def triangular_mu(x, scale):
    y = x.copy() * scale
    y[y > (scale * 1 / 2)] = scale - y[y > (scale * 1 / 2)]
    return y


class NonLinearMu:
    def __init__(self, args):
        if args.func_param == 0:
            self.A = 0.2
            self.B = 0.3
            self.C = 25
            self.D = 7
        elif args.func_param == 1:
            self.A = 0.25
            self.B = 0.25
            self.C = 3
            self.D = 50
        elif args.func_param == 2:
            self.A = 0.25
            self.B = 0.25
            self.C = 15
            self.D = 32
        elif args.func_param == 3:
            self.A = 0.25
            self.B = 0.25
            self.C = 20
            self.D = 23
        elif args.func_param == 4:
            self.A = 0.1
            self.B = 0.9
            self.C = 14
            self.D = 16
        elif args.func_param == 5:
            self.A = 0.1
            self.B = 1 - self.A
            self.C = 30
            self.D = 36
        elif args.func_param == 6:
            self.A = 0.4
            self.B = 1 - self.A
            self.C = 30
            self.D = 33
        elif args.func_param == 7:
            self.A = 0.5
            self.B = 1 - self.A
            self.C = 20
            self.D = 5
        elif args.func_param == 8:
            self.A = 0.5
            self.B = 1 - self.A
            self.C = 40
            self.D = 9
        elif args.func_param == 9:
            self.A = 0.3
            self.B = 1 - self.A
            self.C = 4
            self.D = 15
        elif args.func_param == 10:
            self.A = 0.1
            self.B = 1 - self.A
            self.C = 10
            self.D = 15

    def __call__(self, x, scale):
        y = self.A * np.sin(self.C * x) + self.B * np.sin(self.D * x)
        y = y - y.min()
        return y * scale


def sin(x, scale):
    y = np.sin(x).copy() * scale
    return y * scale


def build_function(args):
    if args.func == "triangular":
        func = triangular_mu
    elif args.func == "sin":
        func = sin
    elif args.func == "non-linear":
        func = NonLinearMu(args)
    return func


class NonLinearMu2:
    def __init__(self, func_param):
        if func_param == 0:
            self.A = 0.4
            self.B = 0.6
            self.C = 25
            self.D = 7
        elif func_param == 1:
            self.A = 0.5
            self.B = 0.5
            self.C = 3
            self.D = 50
        elif func_param == 2:
            self.A = 0.5
            self.B = 0.5
            self.C = 15
            self.D = 32
        elif func_param == 3:
            self.A = 0.5
            self.B = 0.5
            self.C = 10
            self.D = 10
        elif func_param == 4:
            self.A = 0.1
            self.B = 0.9
            self.C = 4
            self.D = 6
        elif func_param == 5:
            self.A = 0.1
            self.B = 1 - self.A
            self.C = 30
            self.D = 36
        elif func_param == 6:
            self.A = 0.4
            self.B = 1 - self.A
            self.C = 30
            self.D = 33
        elif func_param == 7:
            self.A = 0.5
            self.B = 1 - self.A
            self.C = 20
            self.D = 5
        elif func_param == 8:
            self.A = 0.5
            self.B = 1 - self.A
            self.C = 40
            self.D = 9
        elif func_param == 9:
            self.A = 0.3
            self.B = 1 - self.A
            self.C = 4
            self.D = 15
        elif func_param == 10:
            self.A = 0.1
            self.B = 1 - self.A
            self.C = 10
            self.D = 3

    def __call__(self, x, scale):
        y = self.A * np.sin(self.C * x) + self.B * np.sin(self.D * x)
        y = y - y.min()
        return y * scale


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(0, 1, 100)
    scale = 1
    # y = triangular_mu(x, scale)
    custom_mu = NonLinearMu2(4)
    y = custom_mu(x, 1)
    plt.plot(x, y)
    plt.title("Triangular Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("test.png")
