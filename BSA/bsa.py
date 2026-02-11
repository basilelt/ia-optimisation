import numpy as np


def bsa(obj_fun, N, D, maxcycle, low, up):
    """
    Backtracking Search Algorithm (BSA)

    Parameters:
    -----------
    obj_fun : function
        Objective function to minimize
    N : int
        Population size
    D : int
        Problem dimension
    maxcycle : int
        Maximum number of iterations
    low : array-like
        Lower bounds for each dimension
    up : array-like
        Upper bounds for each dimension

    Returns:
    --------
    globalminimum : float
        Best fitness value found
    globalminimizer : array
        Best solution found
    """

    # INITIALIZATION
    globalminimum = np.inf

    # Initialize population P and oldP
    P = np.random.rand(N, D) * (up - low) + low
    oldP = np.random.rand(N, D) * (up - low) + low

    # Calculate initial fitness values
    fitnessP = np.array([obj_fun(P[i]) for i in range(N)])

    # Main loop
    for iteration in range(maxcycle):
        # SELECTION-I
        a = np.random.rand()
        b = np.random.rand()

        if a < b:
            oldP = P.copy()

        # Permuting: arbitrary changes in positions of two individuals in oldP
        oldP = np.random.permutation(oldP)

        # Generation of Trial-Population
        # MUTATION
        mutant = P + 3 * np.random.randn(N, D) * (oldP - P)

        # CROSSOVER
        # Initial-map is an N-by-D matrix of ones
        map1 = np.ones((N, D))

        c = np.random.rand()
        d = np.random.rand()

        if c < d:
            for i in range(N):
                # mixrate determines how many dimensions to mutate
                mixrate = np.random.rand()
                num_to_zero = int(mixrate * D)
                u = np.random.permutation(D)
                map1[i, u[:num_to_zero]] = 0
        else:
            # Random single dimension for each individual
            for i in range(N):
                rand_idx = np.random.randint(D)
                map1[i, rand_idx] = 0

        # Generation of Trial Population, T
        T = mutant.copy()
        for i in range(N):
            for j in range(D):
                if map1[i, j] == 1:
                    T[i, j] = P[i, j]

        # Boundary Control Mechanism
        for i in range(N):
            for j in range(D):
                if T[i, j] < low[j] or T[i, j] > up[j]:
                    T[i, j] = np.random.rand() * (up[j] - low[j]) + low[j]

        # SELECTION-II
        fitnessT = np.array([obj_fun(T[i]) for i in range(N)])

        for i in range(N):
            if fitnessT[i] < fitnessP[i]:
                fitnessP[i] = fitnessT[i]
                P[i] = T[i]

        # Update global minimum
        best_idx = np.argmin(fitnessP)
        fitnessPbest = fitnessP[best_idx]

        if fitnessPbest < globalminimum:
            globalminimum = fitnessPbest
            globalminimizer = P[best_idx].copy()

    return globalminimum, globalminimizer


# Exemple d'utilisation
if __name__ == "__main__":
    # Fonction de test (sphere function)
    def sphere(x):
        return np.sum(x**2)

    # Paramètres
    N = 50  # Taille de la population
    D = 10  # Dimension du problème
    maxcycle = 1000  # Nombre maximum d'itérations
    low = np.full(D, -100)  # Bornes inférieures
    up = np.full(D, 100)  # Bornes supérieures

    # Exécution de l'algorithme
    best_fitness, best_solution = bsa(sphere, N, D, maxcycle, low, up)

    print(f"Meilleur fitness: {best_fitness}")
    print(f"Meilleure solution: {best_solution}")
