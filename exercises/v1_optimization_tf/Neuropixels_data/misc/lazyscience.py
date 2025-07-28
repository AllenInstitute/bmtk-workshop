import numpy as np
from scipy.stats import poisson, nbinom
import h5py
import sys
from iminuit.pdg_format import pdg_format
import iminuit
import iminuit.cost
from numba import jit, njit


def spike_mle_nbinom(counts, estimator, inits, limits=None, do_minos=False):
    # common task for spiking activity.
    # fit overdispersed spikes with negative binomial function!
    # just returns Minuit object.
    varnames = estimator.__code__.co_varnames

    argdict = {}
    argdict["binom_n"] = counts.mean() * 100  # an arbitrary large number...
    # n -> inf is Poisson, so this is starting close to Poisson.
    # argdict = {k: v for k, v in zip(varnames, inits)}
    for k, v in zip(varnames, inits):
        argdict[k] = v

    def nbinom_pmf(counts, binom_n, *est_args):
        return nbinom.pmf(counts, binom_n, binom_n / (estimator(*est_args) + binom_n))

    cost_func = iminuit.cost.UnbinnedNLL(counts, nbinom_pmf)
    # Let's make a fake Namespace for this function.
    fake_func_code = iminuit.util.make_func_code(("binom_n",) + varnames)
    cost_func._func_code = fake_func_code

    m = iminuit.Minuit(cost_func, **argdict)
    m.limits["binom_n"] = (0, None)
    if not (limits == None):
        for v in limits:
            m.limits[v] = limits[v]

    m.migrad()
    if do_minos:
        m.minos()
    return m


def format(vals, errs):
    # apply PDG (Particle Data Group) formatting to arrays
    return [pdg_format(v, e) for v, e in zip(vals, errs)]


def mean_sem(array, axis=0):
    # calculate the mean and sem and return it in a tuple
    mean = array.mean(axis=axis)
    sem = array.std(axis=axis) / np.sqrt(array.shape[axis])
    return (mean, sem)


# IO related
def save(varlist, filename=None, mode="w"):
    # formatting the inputs
    if isinstance(varlist, str):
        varlist = [varlist]  # make it a list so that the following code work
    if filename == None:
        filename = varlist[0] + ".h5"

    # note that this next line is not a very good practice.
    # just being lazy...
    caller = sys._getframe(1)

    # save variables into a file
    with h5py.File(filename, mode) as f:
        for vname in varlist:
            f.create_dataset(vname, data=caller.f_locals[vname])


def load(filename):

    output = {}
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            output[key] = np.array(f[key])

    return output


def quasi_poisson_sig_test_counts(
    response_counts, window_size1, spont_counts, window_size2, unit_axis=-1
):
    arraydim = len(response_counts.shape)
    unit_axis = unit_axis % arraydim  # making it positive value
    sumaxis = tuple(np.array(np.setdiff1d(range(arraydim), unit_axis), dtype=np.int))
    ntrials = response_counts.size / response_counts.shape[unit_axis]
    total_duration = ntrials * window_size1

    # print(sumaxis)
    response_fr = response_counts.mean(axis=sumaxis) / window_size1
    spont_fr = spont_counts.mean(axis=sumaxis) / window_size2
    spont_var = spont_counts.var(axis=sumaxis) / window_size2
    pval = quasi_poisson_sig_test_frvar(
        response_fr, total_duration, spont_fr, spont_var
    )

    return pval


# spiking data analysis
def quasi_poisson_sig_test(
    response_counts, window_size, spont_fr, spont_disper, unit_dim=-1
):
    # if unit_dim is None, it assumes it is the last dimension
    arraydim = len(response_counts.shape)
    unit_dim = unit_dim % arraydim  # making it positive value
    sumdims = np.setdiff1d(range(arraydim), unit_dim)

    # is the windowsize ms or s?
    ntrials = response_counts.size / response_counts.shape[unit_dim]
    expected_counts = spont_fr * window_size * ntrials
    actual_counts = response_counts.sum(dim=sumdims)

    k = actual_counts / spont_disper
    mu = expected_counts / spont_disper
    pval = 1 - poisson.cdf(k, mu)
    return pval


def quasi_poisson_sig_test_fr(response_fr, total_duration, spont_fr, spont_disper):
    expected_counts = spont_fr * total_duration
    actual_counts = response_fr * total_duration

    k = actual_counts / spont_disper
    mu = expected_counts / spont_disper
    pval = 1 - poisson.cdf(k, mu)
    return pval


def quasi_poisson_sig_test_frvar(response_fr, total_duration, spont_fr, spont_var):
    return quasi_poisson_sig_test_fr(
        response_fr, total_duration, spont_fr, spont_var / spont_fr
    )


def center_of_mass(nparray, axis):
    # contains 0, 1, 2, ...
    indexlist = np.array(range(nparray.shape[axis]))
    indexlist = np.swapaxes(indexlist, 0, axis)
    return (nparray * indexlist).sum(axis=axis) / nparray.sum(axis=axis)


def binomial_prob(truth_table, disp=False):
    prob = truth_table.sum() / truth_table.size
    err = np.sqrt(prob * (1 - prob) / truth_table.size)
    if disp:
        print("Binomial probability: " + pdg_format(prob, err))
    return (prob, err)


def cond_prob(truth_table1, truth_table2):
    # calculate conditional probability of two tables
    def efunc(p, n):
        return np.sqrt(p * (1 - p) / n)

    n = np.float64(truth_table1.size)
    n1 = np.float64(np.count_nonzero(truth_table1))
    n2 = np.float64(np.count_nonzero(truth_table2))
    p1 = n1 / n
    p2 = n2 / n
    p1e = efunc(p1, n)
    p2e = efunc(p2, n)
    n12 = np.count_nonzero(truth_table1 * truth_table2)
    p12 = n12 / n
    p12e = efunc(p12, n)

    p1g2 = n12 / n2
    p1g2e = efunc(p1g2, n2)
    p2g1 = n12 / n1
    p2g1e = efunc(p2g1, n1)
    print(
        "p1:%.0f±%.0f%% p2:%.0f±%.0f%% p1g2:%.0f±%.0f%% p2g1:%.0f±%.0f%%\n"
        % (
            p1 * 100,
            p1e * 100,
            p2 * 100,
            p2e * 100,
            p1g2 * 100,
            p1g2e * 100,
            p2g1 * 100,
            p2g1e * 100,
        )
    )

