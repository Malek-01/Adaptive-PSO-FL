import numpy as np

def Griewank(x):
    if len(x.shape) == 1:  # For individual particle
        sum_term = np.sum(x**2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    else:  # For multiple particles
        sum_term = np.sum(x**2, axis=1) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, x.shape[1] + 1))), axis=1)
    return 1 + sum_term - prod_term

def Cigar(x):
    if len(x.shape) == 1:
        return x[0]**2 + 1e6 * np.sum(x[1:]**2)
    else:
        return x[:,0]**2 + 1e6 * np.sum(x[:,1:]**2, axis=1)
    
def Ackley(x):
    if len(x.shape) == 1:
        a = 20
        b = 0.2
        c = 2 * np.pi
        term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / len(x)))
        term2 = -np.exp(np.sum(np.cos(c * x)) / len(x))
        return term1 + term2 + a + np.exp(1)
    else:
        # if not np.logical_and(x >= -32, x <= 32).all():
        #     raise ValueError("Input for Ackley function must be within [-32, 32].")
        d = x.shape[1]
        return (
            -20.0 * np.exp(-0.2 * np.sqrt((1 / d) * (x ** 2).sum(axis=1)))
            - np.exp((1 / float(d)) * np.cos(2 * np.pi * x).sum(axis=1))
            + 20.0
            + np.exp(1)
        )
    
def Discus(x):
    if len(x.shape) == 1:
        return 1e6 * (x[0]**2) + np.sum(x[1:]**2)
    else:
        return 1e6 * (x[:, 0]**2) + np.sum(x[:, 1:]**2, axis=1)

def Rastrigin(x):
    if len(x.shape) == 1:
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    else:
        num_points, num_dimensions = x.shape
        result = np.zeros(num_points)
        for i in range(num_points):
            result[i] = 10 * num_dimensions + np.sum(x[i]**2 - 10 * np.cos(2 * np.pi * x[i]))
        return result

def Sphere(x):
    if len(x.shape) == 1:
        return (x ** 2.0).sum()
    else:
        return (x ** 2.0).sum(axis=1)

def Levi(x):
    if len(x.shape) == 1:
        w_ = 1 + (x - 1) / 4

        return (
            np.sin(np.pi * w_[0]) ** 2.0
            + ((x - 1) ** 2.0).sum()
            * (1 + 10 * np.sin(np.pi * w_.sum() + 1) ** 2.0)
            + (w_[1] - 1) ** 2.0 * (1 + np.sin(2 * np.pi * w_[1]) ** 2.0)
        )
    else:
        mask = np.full(x.shape, False)
        mask[:, -1] = True
        masked_x = np.ma.array(x, mask=mask)

        w_ = 1 + (x - 1) / 4
        masked_w_ = np.ma.array(w_, mask=mask)
        d_ = x.shape[1] - 1

        return (
            np.sin(np.pi * w_[:, 0]) ** 2.0
            + ((masked_x - 1) ** 2.0).sum(axis=1)
            * (1 + 10 * np.sin(np.pi * (masked_w_).sum(axis=1) + 1) ** 2.0)
            + (w_[:, d_] - 1) ** 2.0 * (1 + np.sin(2 * np.pi * w_[:, d_]) ** 2.0)
        )

def Levy(x):
    if len(x.shape) == 1:
        w = 1 + (x - 1) / 4

        return (
            np.sin(np.pi * w[0]) ** 2.0
            + ((x - 1) ** 2.0).sum()
            * (1 + 10 * np.sin(np.pi * w.sum() + 1) ** 2.0)
            + (w[1] - 1) ** 2.0 * (1 + np.sin(2 * np.pi * w[1]) ** 2.0)
        )
    else:
        mask = np.full(x.shape, False)
        mask[:, -1] = True
        masked_x = np.ma.array(x, mask=mask)

        w = 1 + (x - 1) / 4
        masked_w = np.ma.array(w, mask=mask)
        d = x.shape[1] - 1

        return (
            np.sin(np.pi * w[:, 0]) ** 2.0
            + ((masked_x - 1) ** 2.0).sum(axis=1)
            * (1 + 10 * np.sin(np.pi * (masked_w).sum(axis=1) + 1) ** 2.0)
            + (w[:, d] - 1) ** 2.0 * (1 + np.sin(2 * np.pi * w[:, d]) ** 2.0)
        )

def Ellipsoid(x):
    if len(x.shape) == 1:
        return np.sum(np.arange(1, len(x) + 1) * x**2)
    else:
        return np.sum(np.arange(1, x.shape[1] + 1) * x**2, axis=1)

def Schwefel(x):
    if len(x.shape) == 1:
        return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    else:
        return 418.9829 * x.shape[1] - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)

def Schaffer2(x):
    if len(x.shape) == 1:
        x_ = x[0]
        y_ = x[0]
    else:
        x_ = x[:, 0]
        y_ = x[:, 1]
    
    return 0.5 + (
        (np.sin(x_ ** 2.0 - y_ ** 2.0) ** 2.0 - 0.5)
        / ((1 + 0.001 * (x_ ** 2.0 + y_ ** 2.0)) ** 2.0)
    )


def Step(x):
    if len(x.shape) == 1:
        return np.sum(np.floor(x + 0.5) ** 2)
    else:
        return np.sum(np.floor(x + 0.5) ** 2, axis=1)
    # Dictionnaire de fonctions
    
functions_dict = {
    "Griewank": Griewank,
    "Cigar": Cigar,
    "Ackley": Ackley,
    "Discus": Discus,
    "Rastrigin": Rastrigin,
    "Sphere": Sphere,
    "Levi": Levi,
    "Levy": Levy,
    "Ellipsoid": Ellipsoid,
    "Schaffer2": Schaffer2,
    "Step": Step,
}

def Weierstrass(x):
    a = 0.5
    b = 3.0
    kmax = 20

    if len(x.shape) == 1:
        x_ = x[0]
        result = 0.0
        for k in range(kmax + 1):
            result += (a ** k) * np.cos(2 * np.pi * (b ** k) * (x_ + 0.5))
        return result - kmax * np.sum(a ** kmax * np.cos(2 * np.pi * (b ** kmax) * 0.5))
    else:
        num_samples, num_dims = x.shape
        result = np.zeros(num_samples)
        for sample in range(num_samples):
            x_ = x[sample]
            term_sum = 0.0
            for dim in range(num_dims):
                term_sum += (a ** dim) * np.cos(2 * np.pi * (b ** dim) * (x_[dim] + 0.5))
            result[sample] = term_sum - kmax * np.sum(a ** kmax * np.cos(2 * np.pi * (b ** kmax) * 0.5))
        return result




def Schwefel(x):
    if len(x.shape) == 1:
        return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    else:
        return 418.9829 * x.shape[1] - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)
