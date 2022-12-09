"""Perform Monte Carlo simulation to assert that the findings in Proposition 1 remains
in finite-sample setting.

The idea is embedded by the Method simu from Monte_Carlo's object. Here, we generate n_iter
samples of length n_sample (a list of int) where we compute the w-madogram on each. We are able
to compute the empirical variance over these n_iter w-madogram and compare it to the theoretical
one.

Structure :
    - Monte Carlo (:py:class :`Monte_Carlo`)
"""

import numpy as np
import pandas as pd


class MonteCarlo:
    """Base class for Monte-Carlo simulations.

    Inputs
    ------
        n_iter (int)                    : number of Monte Carlo simulation.
        n_sample (list of int or [int]) : multiple length of sample.
        random_seed (Union[int, None])  : seed for the random generator.
        P (array [float])               : dim times dim array of probabilities of presence.
        copula (object)                 : law of the vector of the uniform margin.
        copula_miss (object)            : dependence modeling of the couple (I,J).
        w ([float])                     : vector of weight belonging to the simplex.

    Methods
    -------
                simu (DataFrame)               : results of the Monte-Carlo simulation.

    Examples
    --------
        >>> import base
        >>> import mv_evd
        >>> import monte_carlo
        >>> import utils
        >>> import numpy as np
        >>> from scipy.stats import norm

        >>> dim, n_iter, n_sample = 3, 512, [1024]
        >>> theta, asy = 1/2, [0.7,0.1,0.6,[0.1,0.2],[0.1,0.1],[0.4,0.1],[0.1,0.3,0.2]]
        >>> copula = mv_evd.Asymmetric_logistic(theta = theta,
             dim = dim, asy = asy, n_sample = np.max(n_sample))
        >>> P, p = np.ones([dim,dim]), 1.0 # no missing values
        >>> w = utils.simplex(dim, n = 1)[0]
        >>> Monte = monte_carlo.Monte_Carlo(n_iter = n_iter, n_sample = n_sample,
             w = w, copula = copula, P)

        >>> df_wmado = Monte.finite_sample(norm.ppf, corr = False)
        >>> y = copula.var_mado(w, P, p, corr = False)
        >>> print(y), print(df_wmado['scaled'].var())

        OUT : '0.012860918152289402', '0.012529189725192727'

    """

    n_iter = None
    n_sample = None
    weight = None
    copula = None
    copula_miss = None
    dim = None

    def __init__(self, n_iter=None, n_sample=None, weight=None,
                 copula=None, matp=None, copula_miss=None):
        """
            Initialize Monte_Carlo object
        """

        self.n_iter = n_iter
        if isinstance(n_sample, int):
            self.n_sample = [n_sample]
        else:
            self.n_sample = n_sample
        self.weight = weight
        self.copula = copula
        self.matp = matp
        self.copula_miss = copula_miss
        self.dim = self.copula.dim

    def check_w(self):
        """
            Validate the weights inserted
            Raises :
                ValueError : If w is not in simplex
        """
        if (len(self.weight) != self.dim and np.sum(self.weight) != 1):
            message = "The w value {} is not in the simplex."
            raise ValueError(message.format(self.weight))

    def _wmado(self, mat, miss, corr=True):
        """
            This function computes the w-madogram

            Inputs
            ------
            mat (array([float]) of n_sample \times dim) : a matrix
            w                                           : element of the simplex
            miss (array([int]))                         : list of observed data
            corr (True or False)                        : If true, return
                                                          corrected version of
                                                          w-madogram

            Outputs
            -------
            w-madogram
        """
        nnb = mat.shape[1]
        tnb = mat.shape[0]

        femscale = np.zeros([tnb, nnb])
        cross = np.ones(tnb)
        for j in range(0, nnb):

            cross *= miss[:, j]
            x_vec = np.array(mat[:, j])
            femp = empcdf(x_vec, miss[:, j])
            femscale[:, j] = np.power(femp, 1/self.weight[j])

        femscale *= cross.reshape(tnb, 1)
        if corr is True:
            value_1 = np.amax(femscale, 1)
            value_2 = (1/self.copula.dim) * np.sum(femscale, 1)
            value_3 = (self.dim-1)/self.dim * np.sum(femscale*self.weight, 1)
            return (1/np.sum(cross)) * np.sum(value_1 - value_2 - value_3) + \
                ((self.dim-1)/self.dim) * \
                np.sum(self.weight * self.weight/(1+self.weight))
        value_1 = np.amax(femscale, 1)
        value_2 = (1/self.copula.dim) * np.sum(femscale, 1)
        mado = (1/(np.sum(cross))) * np.sum(value_1 - value_2)

        return mado

    def _gen_missing(self):
        """This function returns an array max(n_sample) times dim of binary indicate missing values.
        Dependence between (I_0, dots, I_{dim-1}) is given by copula_miss. The idea is the following
        I_j follows Ber(P[j][j]) and (I_0, dots ,I_{dim-1}) follows
        Ber(copula_miss(P[0][0], dots, P[dim-1][dim-1])).

        We simulate it by generating a sample (U_0, dots, U_{dim-1}) of shape (n_sample, dim)
        from copula_miss. Then, mat_{i0} = 1 if U_{i1} <= P[0][0], dots, mat_{i(dim-1)} = 1
        if U_{i(dim-1)} <= P[dim-1][dim-1]. These random variables are indeed Bernoulli.

        Also P(I_{0} = 1, dots, I_{dim-1} = 1) =
        P(U_{0} <= P[0][0], dots, U_{dim-1} <= P[dim-1][dim-1]) =
        C(P[0][0],dots, P[dim-1][dim-1])

        Ouputs
        ------
        Array of shape (n_sample, dim) with the ith of the jth column equals 1
        if we observe mat_{ij}, 0 otherwise.
        """
        if self.copula_miss is None:
            return np.array([np.random.binomial(1,
                                                self.matp[j][j], np.max(self.n_sample))
                             for j in range(0, self.dim)]).reshape(np.max(self.n_sample),
                                                                   self.dim)
        sample_ = self.copula_miss.sample_unimargin()
        miss_ = np.array([1 * (sample_[:, j] <= self.matp[j][j])
                         for j in range(0, self.dim)]).reshape(np.max(self.n_sample), self.dim)
        return miss_

    def finite_sample(self, inv_cdf, corr=True):
        """Perform Monte Carlo simulation to obtain empirical counterpart of the wmadogram.

        Inputs
        ------
            inv_cdf : quantile function so that margins follows the given law.
            corr    : if False, returns the hybrid madogram estimator, if yes
                      return its corrected version.

        Output
        ------
            pd.Dataframe : 'wmado', estimator of the w-madogram.
                           'n', length of the sample.
                           'scaled', normalized estimation error.
        """
        output = []
        for miter in range(self.n_iter):
            wmado_store = np.zeros(len(self.n_sample))
            obs_all = self.copula.sample(inv_cdf)
            miss_all = self._gen_missing()
            for i in range(0, len(self.n_sample)):
                obs = obs_all[:self.n_sample[i]]
                miss = miss_all[:self.n_sample[i]]
                wmado = self._wmado(obs, miss, corr)
                wmado_store[i] = wmado

            output_cbind = np.c_[wmado_store, self.n_sample]
            output.append(output_cbind)
        df_wmado = pd.DataFrame(np.concatenate(output))
        df_wmado.columns = ['wmado', 'n']
        df_wmado['scaled'] = (
            df_wmado.wmado - self.copula.true_wmado(self.weight)) * np.sqrt(df_wmado.n)
        return df_wmado


def empcdf(data, miss):
    """Compute uniform ECDF.
    Inputs
    ------
        data (list([float])) : array of observations.
    Output
    ------
        Empirical uniform margin.
    """
    index = np.argsort(data)
    ecdf = np.zeros(len(index))
    for i in index:
        ecdf[i] = (1.0 / np.sum(miss)) * np.sum((data <= data[i]) * miss)
    return ecdf
