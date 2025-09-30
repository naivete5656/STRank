import copy
import numpy as np
from scipy.stats import nbinom


def simple(mu_func, n_samples=1000, r=10, scale=0.5, bias=0.1, sampling="uniform", drop_rate=0):
    np.random.seed(42)

    # # Generate input features
    # x1 = np.linspace(0, 1, n_samples * 10)
    # # sampling
    # x1 = np.random.choice(x1, n_samples, replace=False)
    if "uniform" in sampling:
        x1 = np.random.uniform(0, 1, n_samples)
    elif sampling == "imbalanced":
        # x1 = np.random.uniform(0, 1, round(n_samples - n_samples * 0.8))

        x_mean = np.random.uniform(0, 1)
        x1 = np.random.uniform(max(0, x_mean - 0.1), min(1, x_mean + 0.1), n_samples)
        # x1 = np.concatenate([x1, x2])
        
    # Negative Binomial: Mean = r(1-p)/p, Variance = r(1-p)/p^2
    mu1 = mu_func(x1, scale) + bias
    # Adjust p to match the generated means while keeping r constant
    p1 = r / (r + mu1)
    # Generate Negative Binomial samples
    y1 = np.array([nbinom.rvs(n=r, p=p_val) for p_val in p1])

    probs = 1 - np.ones_like(y1) * drop_rate
    # down_count = torch.distributions.Binomial(counts, probs).sample()
    y1 = np.random.binomial(n=y1, p=probs)

    return x1, y1


def multi_pat(custom_mu, args):
    args_copy = copy.deepcopy(args)
    args_copy.sampling = "uniform"
    x_list, y_list, gl_list = [], [], []
    for n_pat in range(args.num_pat):
        x, y = simple(
            custom_mu,
            args_copy.n_sample,
            args_copy.r,
            args_copy.scale,
            args_copy.bias,
            args_copy.sampling,
        )

        args_copy.n_sample = args.n_sample
        args_copy.scale = args.scale * 10
        args_copy.bias = 10
        args_copy.sampling = args.sampling
        # args_copy.sampling = "imbalanced"

        x_list.append(x)
        y_list.append(y)
        gl_list.append(np.full_like(x, n_pat))
    x_train = np.concatenate(x_list)
    y_train = np.concatenate(y_list)
    gl_train = np.concatenate(gl_list)
    return x_train, y_train, gl_train


def high_expression_toy_data(n_samples=1000, mu_func=None, r=10, beta=0.5, shift=30):
    """
    Generate toy data for regression task with two slightly different Negative Binomial Distributions

    Parameters:
    - n_samples: Number of samples to generate
    - mu_func: Function to generate mean (μ)
    - r: Number of failures before the experiment is stopped (dispersion parameter)
    - p: Probability of success in each trial
    - beta: Shift parameter between two distributions

    Returns:
    - X: Input features
    - y1: Targets for first distribution
    - y2: Targets for second distribution
    """
    np.random.seed(42)

    # Generate input features
    x1 = np.linspace(0, 10, n_samples * 10)
    x1 = np.random.choice(x1, n_samples, replace=False)
    mu1 = mu_func(x1) + shift
    p1 = r / (r + mu1)
    y1 = np.array([nbinom.rvs(n=r, p=p_val) for p_val in p1])
    bias1 = shift

    x2 = np.linspace(0, 4, n_samples * 10)
    x2 = np.random.choice(x2, n_samples, replace=False)
    mu2 = mu_func(x2) + shift + beta * 50
    p2 = r / (r + mu2)
    y2 = np.array([nbinom.rvs(n=r, p=p_val) for p_val in p2])
    bias2 = shift + beta * 50

    x3 = np.linspace(0, 10, n_samples)
    mu3 = mu_func(x3) + beta * 50
    p3 = r / (r + mu3)
    y3 = np.array([nbinom.rvs(n=r, p=p_val) for p_val in p3])

    return x1, y1, x2, y2, x3, y3, bias1, bias2, mu3


def additive_scaling(n_samples=1000, mu_func=None, r=10, beta=0.5, shift=1):
    """
    Generate toy data for regression task with two slightly different Negative Binomial Distributions

    Parameters:
    - n_samples: Number of samples to generate
    - mu_func: Function to generate mean (μ)
    - r: Number of failures before the experiment is stopped (dispersion parameter)
    - p: Probability of success in each trial
    - beta: Shift parameter between two distributions

    Returns:
    - X: Input features
    - y1: Targets for first distribution
    - y2: Targets for second distribution
    """
    np.random.seed(42)

    # Generate input features
    x1 = np.linspace(0, 0.2, n_samples * 10)
    x1 = np.random.choice(x1, n_samples, replace=False)
    mu1 = mu_func(x1) + shift * 25
    p1 = r / (r + mu1)
    y1 = np.array([nbinom.rvs(n=r, p=p_val) for p_val in p1])
    bias1 = shift * 25

    x2 = np.linspace(0, 1, n_samples * 10)
    x2 = np.random.choice(x2, n_samples, replace=False)
    mu2 = mu_func(x2)
    p2 = r / (r + mu2)
    y2 = np.array([nbinom.rvs(n=r, p=p_val) for p_val in p2])
    # y2 = np.clip(y2 - 102.5, 0, None)
    bias2 = 0

    x3 = np.linspace(0, 1, n_samples)
    mu3 = mu_func(x3) + beta * 50
    p3 = r / (r + mu3)
    y3 = np.array([nbinom.rvs(n=r, p=p_val) for p_val in p3])

    return x1, y1, x2, y2, x3, y3, bias1, bias2, mu3


def scaling(n_samples=1000, mu_func=None, alpha=1, r=10, beta=0.5, shift=1):
    """
    Generate toy data for regression task with two slightly different Negative Binomial Distributions

    Parameters:
    - n_samples: Number of samples to generate
    - mu_func: Function to generate mean (μ)
    - r: Number of failures before the experiment is stopped (dispersion parameter)
    - p: Probability of success in each trial
    - beta: Shift parameter between two distributions

    Returns:
    - X: Input features
    - y1: Targets for first distribution
    - y2: Targets for second distribution
    """
    np.random.seed(42)

    # Generate input features
    x1 = np.linspace(0, 0.2, n_samples * 10)
    x1 = np.random.choice(x1, n_samples, replace=False)
    mu1 = mu_func(x1)
    p1 = r / (r + mu1)
    y1 = np.array([nbinom.rvs(n=r, p=p_val) for p_val in p1])

    x2 = np.linspace(0, 1, n_samples * 10)
    x2 = np.random.choice(x2, n_samples, replace=False)
    mu2 = mu_func(x2) / alpha
    p2 = r / (r + mu2)
    y2 = np.array([nbinom.rvs(n=r, p=p_val) for p_val in p2])

    x3 = np.linspace(0, 1, n_samples)
    mu3 = mu_func(x3) + beta * 50
    p3 = r / (r + mu3)
    y3 = np.array([nbinom.rvs(n=r, p=p_val) for p_val in p3])

    return x1, y1, x2, y2, x3, y3, 1, 1 / alpha, mu3


def param_eval(n_samples=1000, mu_func=None, r=10):
    np.random.seed(42)

    # Generate input features
    x1 = np.linspace(0, 1, n_samples * 10)
    x1 = np.random.choice(x1, n_samples, replace=False)
    mu1 = mu_func(x1) + 4
    p1 = r / (r + mu1)
    y1 = np.array([nbinom.rvs(n=r, p=p_val) for p_val in p1])

    x3 = np.linspace(0, 1, n_samples)
    mu3 = mu_func(x3) + 50
    p3 = r / (r + mu3)
    y3 = np.array([nbinom.rvs(n=r, p=p_val) for p_val in p3])

    return x1, y1, x3, y3, mu3


def generate_toy_data(custom_mu, args):
    if args.data_type == "single":
        x_train, y_train = simple(
            custom_mu, args.n_sample, args.r, args.scale, args.bias, args.sampling, args.drop_rate
        )
        gl_train = np.zeros_like(x_train)
    elif args.data_type == "multi":
        x_train, y_train, gl_train = multi_pat(custom_mu, args)

    x_val = np.random.uniform(0, 1, 10000)
    y_val = custom_mu(x_val, 1) + 0

    x_test = np.random.uniform(0, 1, 10000)
    # x_test = np.linspace(0, 1, 10000)
    y_test = custom_mu(x_test, 1) + 0

    return x_train, y_train, gl_train, x_val, y_val, x_test, y_test


import argparse


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument("--scale", help="data scale", default=0.5, type=float)
    parser.add_argument("--bias", help="data bias", default=1.5, type=float)
    parser.add_argument("--r", help="data dispersion", default=10, type=float)
    parser.add_argument("--func", help="function_type", default="non-linear", type=str)
    parser.add_argument(
        "--data_type", default="multi", choices=["single", "multi"], type=str
    )
    parser.add_argument(
        "--sampling", default="uniform", choices=["uniform", "imbalanced"], type=str
    )
    parser.add_argument("--num_pat", help="number of patient", default=2, type=int)
    parser.add_argument("--func_param", help="function pattern", default=0, type=int)
    parser.add_argument("--n_sample", help="number of sample", default=1000, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    from custom_functions import build_function

    args = parse_args()
    custom_mu = build_function(args)
    generate_toy_data(custom_mu, args)
    print(1)
