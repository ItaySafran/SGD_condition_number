import pickle
import numpy as np
from matplotlib import pyplot as plt


# List of parameters used:
# n - Number of functions whose average is target objective F.
# k - Total number of epochs.
# G - Gradient bound as defined in Assumptions 1 & 2.
# lam - Strong convexity parameter of the objective function
# lam_max - Gradient Lipschitz bound of the objective F which is equivalent to the largest eigenvalue of the matrix A
# setup - A parameter specifying the construction of F used, which is either Eq. (1) in the paper (setup='ss') or
# Eq. (2) (setup='rr').


# Run with-replacement SGD with given parameters on the problem defined in Eq. (1) in the paper (setup='ss') or on the
# problem in Eq. (2) (setup='rr').
def with_replacement(n, k, G, lam, lam_max, setup='ss'):
    eta = np.log(n * k) / (lam * n * k)  # Choose step size according to Theorems 3 & 4 in the paper.
    x1 = -G/lam * 0.5
    x2 = -G/lam_max * 0.5
    for t in range(k):
        for i in range(n):
            x1 = (1 - eta * lam) * x1
            random_bit = np.random.randint(0, 2)  # Choose a bit uniformly at random to choose step type.
            if setup == 'ss':
                x2 = (1 - eta * lam_max) * x2 + eta * G / 2 * (random_bit * 2 - 1)
            else:
                x2 = (1 - eta * lam_max * random_bit) * x2 + eta * G / 2 * (random_bit * 2 - 1)
    # Compute and return loss according to function definition
    if setup == 'ss':
        return lam / 2 * x1 ** 2 + lam_max / 2 * x2 ** 2
    else:
        return lam / 2 * x1 ** 2 + lam_max / 4 * x2 ** 2


# Run without-replacement SGD using single shuffling sampling where a permutation of the functions is selected and then
# used repeatedly throughout all epochs. The algorithm uses the given parameters on the problem defined in Eq. (1) in
# the paper (setup='ss') or on the problem in Eq. (2) (setup='rr').
def SS(n, k, G, lam, lam_max, setup='ss'):
    elements = np.random.permutation(np.append(np.ones(np.floor_divide(n, 2)), -np.ones(np.floor_divide(n, 2))))
    # elements now stores a random permutation of n/2 -1's and n/2 1's, selected uniformly at random.
    eta = np.log(n*k)/(lam * n * k)  # Choose step size according to Theorems 3 & 4 in the paper.
    x1 = -G/lam * 0.5
    x2 = -G/lam_max * 0.5
    # Initialize from a point (x1, x2) satisfying the initialization condition in Assumptions 1 & 2.
    for t in range(k):
        for i in range(n):
            x1 = (1 - eta * lam) * x1
            if setup == 'ss':
                x2 = (1 - eta * lam_max) * x2 + eta * G / 2 * elements[i]
            else:
                x2 = (1 - eta * lam_max * (-0.5 * elements[i] + 0.5)) * x2 + eta * G / 2 * elements[i]
        # Compute and return loss according to function definition
    if setup == 'ss':
        return lam / 2 * x1 ** 2 + lam_max / 2 * x2 ** 2
    else:
        return lam / 2 * x1 ** 2 + lam_max / 4 * x2 ** 2


# Run without-replacement SGD using random reshuffling sampling where a fresh permutation of the functions is selected
# at the start of each epoch. The algorithm uses the given parameters on the problem defined in Eq. (1) in the paper
# (setup='ss') or on the problem in Eq. (2) (setup='rr').
def RR(n, k, G, lam, lam_max, setup='rr'):
    elements = np.append(np.ones(np.floor_divide(n, 2)), -np.ones(np.floor_divide(n, 2)))
    # elements stores an array of n/2 1's followed by n/2 -1's.
    eta = np.log(n*k)/(lam * n * k)
    x1 = -G/lam * 0.5
    x2 = -G/lam_max * 0.5
    for t in range(k):
        elements = np.random.permutation(elements)
        # elements now stores a fresh random permutation of -1's and 1's, selected uniformly at random.
        for i in range(n):
            x1 = (1 - eta * lam) * x1
            if setup == 'ss':
                x2 = (1 - eta * lam_max) * x2 + eta * G / 2 * elements[i]
            else:
                x2 = (1 - eta * lam_max * (-0.5 * elements[i] + 0.5)) * x2 + eta * G / 2 * elements[i]
    # Compute and return loss according to function definition
    if setup == 'ss':
        return lam / 2 * x1 ** 2 + lam_max / 2 * x2 ** 2
    else:
        return lam / 2 * x1 ** 2 + lam_max / 4 * x2 ** 2


# Run 'times'-many instantiations of SGD for all three sampling methods, and average over the results. Stores the
# results in a pickled file for each value of k and construction type (E.g. Eq. (1) or Eq. (2) in the paper).
def run_experiment(times, n, ks, G, lam, lam_max, setup='ss'):
    for i in range(len(ks)):
        try:
            result = pickle.load(open(setup + " k=" + str(ks[i]) + ".p", "rb"))
            print("Loaded file '" + str(setup) + " k=" + str(ks[i]) + ".p' containing " + str(len(result[0])) + " instantiations of SGD")
        except (OSError, IOError) as e:
            result = [np.array([]), np.array([]), np.array([])]
            pickle.dump(result, open(str(setup) + " k=" + str(ks[i]) + ".p", "wb"))
            print("Created file '" + str(setup) + " k=" + str(ks[i]) + ".p'")
    for t in range(times):
        print_flag = False
        for i in range(len(ks)):
            result = pickle.load(open(setup + " k=" + str(ks[i]) + ".p", "rb"))
            if len(result[0]) <= t:
                print_flag = True
                result[0] = np.append(result[0], with_replacement(n, ks[i], G, lam, lam_max, setup))
                result[1] = np.append(result[1], SS(n, ks[i], G, lam, lam_max, setup))
                result[2] = np.append(result[2], RR(n, ks[i], G, lam, lam_max, setup))
                pickle.dump(result, open(str(setup) + " k=" + str(ks[i]) + ".p", "wb"))
        if print_flag:
            print('Computed ' + str(t + 1) + ' instantiations of SGD')


# Extract the results of the given values of k and problem setup (E.g. Eq. (1) or Eq. (2) in the paper) and stores the
# results in an array
def get_results(ks, setup):
    wr = np.zeros((len(ks),))
    ss = np.zeros((len(ks),))
    rr = np.zeros((len(ks),))
    wr_ci = np.zeros((len(ks),))
    ss_ci = np.zeros((len(ks),))
    rr_ci = np.zeros((len(ks),))
    for i in range(len(ks)):
        try:
            result = pickle.load(open(setup + " k=" + str(ks[i]) + ".p", "rb"))
        except (OSError, IOError) as e:
            print("Could not open file '" + setup + " k=" + str(ks[i]) + ".p'. File may not exist")
            return 1
        wr[i] = np.mean(np.log10(result[0]))
        ss[i] = np.mean(np.log10(result[1]))
        rr[i] = np.mean(np.log10(result[2]))
        wr_ci[i] = np.std(np.log10(result[0]))
        ss_ci[i] = np.std(np.log10(result[1]))
        rr_ci[i] = np.std(np.log10(result[2]))
    return wr, ss, rr, wr_ci, ss_ci, rr_ci
    # Returns the average loss for with-replacement (wr), single shuffling (ss) and random reshuffling (rr).


# Plots the graph given in Figure 1 in the paper
def plot_graph(ks):
    fig, axs = plt.subplots(2, 1)

    [wr, ss, rr, wr_ci, ss_ci, rr_ci] = get_results(ks, setup='ss')
    axs[0].plot(ks, wr, 'bo', label='With-replacement')
    axs[0].fill_between(ks, (wr - wr_ci), (wr + wr_ci), color='b', alpha=.1)
    axs[0].plot(ks, ss, 'r+', label='Single Shuffling', markersize=7, markeredgewidth=2)
    axs[0].fill_between(ks, (ss - ss_ci), (ss + ss_ci), color='r', alpha=.1)
    axs[0].plot(ks, rr, 'gv', label='Random Reshuffling')
    axs[0].fill_between(ks, (rr - rr_ci), (rr + rr_ci), color='g', alpha=.1)
    axs[0].set_ylabel("Mean Log-Loss", fontsize=16)
    axs[0].legend()
    axs[0].set_yscale('linear')
    axs[0].set_xscale('log')

    [wr, ss, rr, wr_ci, ss_ci, rr_ci] = get_results(ks, setup='rr')
    axs[1].plot(ks, wr, 'bo', label='With-replacement')
    axs[1].fill_between(ks, (wr - wr_ci), (wr + wr_ci), color='b', alpha=.1)
    axs[1].plot(ks, ss, 'r+', label='Single Shuffling', markersize=7, markeredgewidth=2)
    axs[1].fill_between(ks, (ss - ss_ci), (ss + ss_ci), color='r', alpha=.1)
    axs[1].plot(ks, rr, 'gv', label='Random Reshuffling')
    axs[1].fill_between(ks, (rr - rr_ci), (rr + rr_ci), color='g', alpha=.1)
    axs[1].set_ylabel("Mean Log-Loss", fontsize=16)
    axs[1].set_xlabel("k", fontsize=16)
    axs[1].legend()
    axs[1].set_yscale('linear')
    axs[1].set_xscale('log')

    fig.set_size_inches(9, 9)
    plt.show(fig)


# Runs the experiment setup as described in the paper in Sec. 5. The values of k below were chosen so that the graph has
# roughly equal density when the x axis is presented in log scale. To quickly run an experiment, change the value of
# 'times' which controls the number of SGD instantiations to average over, to a smaller number.

k_values = np.append(np.append(np.append(np.append(np.append(range(40, 49, 2), range(50, 121, 4)), range(120, 251, 10)),
                                         range(250, 501, 20)), range(500, 1001, 40)), range(1000, 2001, 80))
run_experiment(times=100, n=500, ks=k_values, G=1, lam=1, lam_max=200, setup='ss')
run_experiment(times=100, n=500, ks=k_values, G=1, lam=1, lam_max=200, setup='rr')
plot_graph(ks=k_values)
