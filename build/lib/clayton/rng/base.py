"""Base module contains method for sampling from a multivariate extreme value copula and
to compute the asymptotic variance of the w-madogram with missing or complete data.

A multivariate copula $C : [0,1]^d \rightarrow [0,1]$ of a d-dimensional random vector $X$ allows
us to separate the effect of dependence from the effect of the marginal distributions. The
copula function completely chracterizes the stochastic dependence between the margins of $X$.
Extreme value copulas are characterized by the stable tail dependence function which the restriction
to the unit simplex is called Pickands dependence function.

Structure :

- Multivariate copula (:py:class:`Multivariate`)
    - Extreme value copula (:py:class:`Extreme`)
"""

import math
import abc
from enum import Enum
import numpy as np
from scipy.optimize import brentq


EPSILON = 1e-12


class CopulaTypes(Enum):
    """
        Available multivariate copula
    """

    CLAYTON = 1
    AMH = 3
    GUMBEL = 4
    FRANK = 5
    JOE = 6
    NELSEN9 = 9
    NELSEN10 = 10
    NELSEN11 = 11
    NELSEN12 = 12
    NELSEN13 = 13
    NELSEN14 = 14
    NELSEN15 = 15
    NELSEN22 = 22
    HUSLER_REISS = 23
    ASYMMETRIC_LOGISTIC = 24
    ASYMMETRIC_NEGATIVE_LOGISTIC = 25
    ASYMMETRIC_MIXED_MODEL = 26
    TEV = 27
    GAUSSIAN = 28
    STUDENT = 29
    DIRICHLET = 30
    BILOG = 31


class Multivariate:
    """Base class for multivariate copulas.
    This class allows to instantiate all its subclasses and serves
    as a unique entry point for the multivariate copulas classes.
    It permit also to compute the variance of the w-madogram for a given point.

    Attributes
    ----------
        copula_type(CopulaTypes)   : see CopulaTypes class.
        dim(int)                   : dimension.
        theta_interval(list[float]): interval of valid theta for the given copula family
        invalid_thetas(list[float]): values, that even though they belong to
                                      :attr: `theta_interval`, shouldn't be considered as valid.
        theta(list[float])         : parameter for the parametric copula
        var_mado(list[float])      : value of the theoretical variance for a given
                                     point in the simplex
        sigmat                     : covariance matrix (only for elliptical, Husler-Reiss).
        asy(list[float])           : asymmetry for the Asy. Log. model.
        psi1, psi2(float)          : asymmetry for the Asy. Log. model when dim = 2.

    Methods
    -------
        sample(np.array([float])): array of shape n_sample x dim of the desired
                                   multivariate copula model where the margins are
                                   inverted by the specified generalized inverse
                                   of a cdf.
    """
    copula_type = None
    theta_interval = []
    invalid_thetas = []
    n_sample = []
    psi1 = 0.0
    psi2 = []
    asy = []
    theta = []
    dim = None
    sigmat = None

    def __init__(self, theta=None, n_sample=None, asy=None,
                 psi1=0.0, psi2=None, dim=2, sigmat=None):
        """Initialize Multivariate object.

        Inputs
        ------
            copula_type (copula_type or st) : subtype of the copula
            theta (list[float] or None)     : list of parameters for the parametric copula
            asy (list[float] or None)       : list of asymetric values for the copula
            n_sample (int or None)          : number of sampled observation
            dim (int or None)                 : dimension
        """
        self.theta = theta
        self.n_sample = n_sample
        self.asy = asy
        self.psi1 = psi1
        self.psi2 = psi2
        self.dim = dim
        self.sigmat = sigmat

    @abc.abstractmethod
    def child_method(self):
        """abstract method
        """

    def sample_unimargin(self):
        """see the corresponding documentation in lower subclasses
        """
        return self.child_method()

    def check_theta(self):
        """Validate the theta inserted.

        Raises
        ------
            ValueError : If there is not in :attr:`theta_interval` or is in :attr:`invalid_thetas`.
        """
        lower, upper = self.theta_interval
        if (not lower <= self.theta <= upper) or (self.theta in self.invalid_thetas):
            message = "The inserted theta value {} is out of limits for the given {} copula."
            raise ValueError(message.format(self.theta, self.copula_type.name))

    def _generate_randomness(self):
        """Generate a bivariate sample draw identically and
        independently from a uniform over the segment [0,1].

        Output
        ------
            output(np.array([float]) shape n_sample x 2): a n_sample x 2 array with each
                                                          component sampled from the desired
                                                          copula under the unit interval.
        """
        v_1 = np.random.uniform(low=0.0, high=1.0, size=self.n_sample)
        v_2 = np.random.uniform(low=0.0, high=1.0, size=self.n_sample)
        output = np.vstack([v_1, v_2]).T
        return output

    def sample(self, inv_cdf):
        """Draws a bivariate sample the desired copula and invert it by
        a given generalized inverse of cumulative distribution function

        Inputs
        ------
            inv_cdf : generalized inverse of cumulative distribution function

        Output
        ------
            output (np.array([float]) with sape n_sample x dim) : sample where the margins
                                                                are specified inv_cdf.
        """
        if isinstance(inv_cdf, list) is False:
            message = "inv_cdf should be a list"
            raise ValueError(message)
        if len(inv_cdf) == 1:
            inv_cdf = np.repeat(inv_cdf, self.dim)
        elif len(inv_cdf) == self.dim:
            pass
        else:
            message = "inv_cdf should be a list of length 1 or {}"
            raise ValueError(message.format(self.dim))
        sample_ = self.sample_unimargin()
        output = np.array([inv_cdf[j](sample_[:, j])
                          for j in range(0, self.dim)])
        output = np.ravel(output).reshape(self.n_sample, self.dim, order='F')
        return output


class Archimedean(Multivariate):
    """Base class for multivariate archimedean copulas.
    This class allowd to use methods which use the generator function.

    Methods
    -------
        sample_uni (np.array([float])) : sample form the multivariate archimedean copula.
                                         Margins are uniform on [0,1].
    """

    @abc.abstractmethod
    def child_method(self):
        """abstract method
        """

    def _generator(self, var):
        """See the documentation in archimedean.py
        """
        return self.child_method()

    def _generator_dot(self, var):
        """See the documentation in archimedean.py
        """
        return self.child_method()

    def _generator_inv(self, var):
        """See the documentation in archimedean.py
        """
        return self.child_method()

    def rfrailty(self):
        """See the documentation in archimedean.py
        """
        return self.child_method()

    def _c(self, var):
        """Return the value of the copula taken on (u,v)
        .. math:: C(u,v) = phi^leftarrow (phi(u) + phi(v)), 0<u,v<1
        """
        value_ = self._generator_inv(np.sum(self._generator(var)))
        return value_

    def _cond_sim(self):
        """Perform conditional simulation. Only useful for Archimedean copulas
        where the frailty distribution is still unknown.

        CopulaTypes
        -----------
            NELSEN9
            NELSEN10
            NELSEN11
            NELSEN12
            NELSEN13
            NELSEN14
            NELSEN15
            NELSEN22
        """
        self.check_theta()
        if self.dim == 2:
            output = np.zeros((self.n_sample, self.dim))
        else:
            message = "This generator can't generate an Archimedean copula for dim greater than 2"
            raise ValueError(message)
        randomness = self._generate_randomness()

        def func(var):
            value_ = (var - self._generator(var) /
                      self._generator_dot(var)) - vectv[1]
            return value_
        for i in range(0, self.n_sample):
            vectv = randomness[i]

            if func(EPSILON) > 0.0:
                sol = 0.0
            else:
                sol = brentq(func, EPSILON, 1-EPSILON)
            vectu = [self._generator_inv(vectv[0] * self._generator(sol)),
                self._generator_inv((1-vectv[0])*self._generator(sol))]
            output[i, :] = vectu
        return output

    def _frailty_sim(self):
        """Sample from Archimedean copula using algorithm where the frailty
        distribution is known.

        CopulaTypes
        -----------
            CLAYTON
            AMH
            JOE
            FRANK
        """
        if self.dim > 2:
            if self.theta < 0:
                message = "The inserted theta value {} is out of limits for the given {} copula. In dimension greater than 2, positive association are only allowed."
                raise ValueError(message.format(
                    self.theta, self.copula_type.name))
        self.check_theta()
        output = np.zeros((self.n_sample, self.dim))
        for i in range(0, self.n_sample):
            samplegamma = np.random.gamma(1, 1, self.dim)
            samplefrailty = self.rfrailty()
            geninv = self._generator_inv(samplegamma/samplefrailty)
            output[i, :] = geninv
        return output

    def sample_unimargin(self):
        """Sample from Archimedean copula with uniform margins.
        Performs different algorithm if the frailty distribution of the
        chosen Archimedean copula is known or not.

        Output
        ------
            output (np.array([float]) with shape n_sample x dim) : sample from the desired
                                                                 copula with uniform margins.

        RaiseValueError
        ---------------
            generator function can't generate Archimedean copula for dim > 2

        """
        output = []
        condsim_numbers = [CopulaTypes.NELSEN9, CopulaTypes.NELSEN10, CopulaTypes.NELSEN11,
                           CopulaTypes.NELSEN12, CopulaTypes.NELSEN13, CopulaTypes.NELSEN14,
                           CopulaTypes.NELSEN15, CopulaTypes.NELSEN22]
        frailty_numbers = [CopulaTypes.FRANK, CopulaTypes.AMH,
                           CopulaTypes.JOE, CopulaTypes.CLAYTON]
        if self.copula_type in condsim_numbers:
            output = self._cond_sim()
        if self.copula_type in frailty_numbers:
            if (self.dim == 2) and (self.theta < 0):
                output = self._cond_sim()
            else:
                output = self._frailty_sim()
        return output


class Extreme(Multivariate):
    """Base class for multivariate extreme value copulas.
    This class allows to use methods which use the Pickands dependence function.

    Methods
    -------
        sample_uni (np.array([float])) : sample from the desired multivariate copula model.
                                         Margins are uniform on [0,1].
        var_FMado (float)              : gives the asymptotic variance of w-madogram for a
                                         multivariate extreme value copula.

    Examples
    --------
        >>> import base
        >>> import mv_evd
        >>> import matplotlib.pyplot as plt

        >>> copula = mv_evd.Logistic(theta = 0.5, dim = 3, n_sample = 1024)
        >>> sample = copula.sample_uni()

        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection = '3d')
        >>> ax.scatter3D(sample[:,0],sample[:,1],sample[:,2], c = 'lightblue', s = 1.0, alpha = 0.5)
        >>> plt.show()
    """

    @abc.abstractmethod
    def child_method(self):
        """abstract method
        """

    def _pickands(self, var):
        """See the documentation in evd.py
        """
        return self.child_method()

    def _pickandsdot(self, var, j):
        """See the documentation in evd.py
        """
        return self.child_method()

    def mvalog_check(self, dep):
        """See the documentation in evd.py
        """
        return self.child_method()

    def rmvlog_tawn(self):
        """See the documentation in evd.py
        """
        return self.child_method()

    def rmvalog_tawn(self, number, alpha, asy):
        """See the documentation in evd.py
        """
        return self.child_method()

    def rextfunc(self, index, cholesky=None):
        """See the documentation in evd.py
        """
        return self.child_method()

    def sigma2covar(self, index):
        """See the documentation in evd.py
        """
        return self.child_method()

    def _l(self, var):
        """Return the value of the stable tail dependence function on u.
        Pickands is parametrize as A(w_0, dots, w_{d-1}) with w_0 = 1-sum_{j=1}^{d-1} w_j

        Inputs
        ------
            u (list[float]) : dim-list with each components between 0 and 1.
        """
        sumu = np.sum(var)
        vectw = var / sumu
        value_ = sumu*self._pickands(vectw)
        return value_

    def _c(self, var):
        """Return the value of the copula taken on u
        .. math:: C(u) = exp(-l(-log(u_1), dots, -log(u_d))), u in [0,1]^d.

        Inputs
        ------
            u (list[float]) : dim-list of float between 0 and 1.
        """
        log_u_ = np.log(var)
        value_ = math.exp(-self._l(-log_u_))
        return value_

    def _mu(self, var, j):
        """Return the value of the jth partial derivative of l.
        ..math:: dot{l}_j(u), u in ]0,1[^d, j in {0,dots,d-1}.

        Inputs
        ------
            var (list[float]) : list of float between 0 and 1.
            j (int)         : jth derivative of the stable tail dependence function.
        """
        sumu = np.sum(var)
        vectw = var / sumu
        if j == 0:
            deriv_ = []
            for i in range(1, self.dim):
                value_deriv = self._pickandsdot(vectw, i) * vectw[i]
                deriv_.append(value_deriv)
            value_ = self._pickands(vectw) - np.sum(deriv_)
        else:
            deriv_ = []
            for i in range(1, self.dim):
                if i == j:
                    value_deriv = -(1-vectw[i]) * self._pickandsdot(vectw, i)
                    deriv_.append(value_deriv)
                else:
                    value_deriv = self._pickandsdot(vectw, i) * vectw[i]
                    deriv_.append(value_deriv)
            value_ = self._pickands(vectw) - np.sum(deriv_)
        return value_

    def _dot_c(self, arg, j):
        """Return the value of dot{C}_j taken on u.
        .. math:: dot{C}_j = C(u)/u_j * _mu_j(u), u in [0,1]^d, j in {0 , dots, d-1}.
        """
        value_ = (self._c(arg) / arg[j]) * self._mu(-np.log(arg), j)
        return value_

    def _cond_sim(self):
        """Draw a bivariate sample from an extreme value copula using conditional simulation.
        Margins are uniform.

        CopulaTypes
        -----------
            ASYMMETRIC_MIXED
            ASYMMETRIC_NEGATIVE_LOGISTIC
        """
        self.check_theta()
        if self.dim > 2:
            message = "The dimension {} inserted is not compatible with {}."
            raise ValueError(message.format(self.dim, self.copula_type.name))
        output = np.zeros((self.n_sample, self.dim))
        randomness = self._generate_randomness()

        def func(var):
            vectu = np.array([vectv[0], var])
            value_ = self._dot_c(vectu, 0) - vectv[1]
            return value_
        for i in range(0, self.n_sample):
            vectv = randomness[i]
            sol = brentq(func, EPSILON, 1-EPSILON)
            vectu = [vectv[0], sol]
            output[i, :] = vectu
        return output

    def _ext_sim(self):
        """ Multivariate extreme value distribution sampling algorithm via extremal functions.
        See Dombry et al [2016], exact simulation of max-stable process for more details.

        CopulaTypes
        -----------
            HUSLER_REISS
            TEV
            DIRICHLET
        """
        if self.copula_type == CopulaTypes.TEV:
            stdev = np.exp(0.5 * np.log(np.diag(self.sigmat)))
            stdevmat = np.linalg.inv(np.diag(stdev))
            self.sigmat = stdevmat @ self.sigmat @ stdevmat

        output = np.zeros((self.n_sample, self.dim))
        matsim = [CopulaTypes.HUSLER_REISS, CopulaTypes.TEV]
        dirlog = [CopulaTypes.DIRICHLET, CopulaTypes.BILOG]
        for i in range(0, self.n_sample):
            zeta = np.random.exponential(1)
            if self.copula_type in matsim:
                covar = self.sigma2covar(0)
                cholesky = np.linalg.cholesky(covar).T
                extfunc = self.rextfunc(0, cholesky)
            if self.copula_type in dirlog:
                extfunc = self.rextfunc(0)

            output[i, :] = extfunc / zeta

            for j in range(1, self.dim):
                zeta = np.random.exponential(1)
                if self.copula_type in matsim:
                    covar = self.sigma2covar(j)
                    cholesky = np.linalg.cholesky(covar).T

                while (1.0 / zeta > output[i, j]):
                    if self.copula_type in matsim:
                        extfunc = self.rextfunc(j, cholesky)
                    if self.copula_type in dirlog:
                        extfunc = self.rextfunc(j)
                    res = True
                    for k in range(0, j):
                        if (extfunc[k] / zeta >= output[i, k]):
                            res = False

                    if res:
                        output[i, :] = np.maximum(output[i, :], extfunc / zeta)
                    zeta += np.random.exponential(1)

        return output

    def sample_unimargin(self):
        """Sample from EV copula with uniform margins.
        Performs different algorithms if a fast random number generator
        is known.

        Output
        ------
            output (np.array([float]) with shape n_sample x dim) : sample from the desired
                                                                 copula with uniform margins.

        RaiseValueError
        ---------------
            generator function can't generate Archimedean copula for dim > 2
        """
        output = []
        condsim_numbers = [CopulaTypes.ASYMMETRIC_NEGATIVE_LOGISTIC,
                           CopulaTypes.ASYMMETRIC_MIXED_MODEL,
                           CopulaTypes.ASYMMETRIC_NEGATIVE_LOGISTIC]
        extsim_numbers = [CopulaTypes.HUSLER_REISS,
                          CopulaTypes.TEV, CopulaTypes.DIRICHLET, CopulaTypes.BILOG]
        if self.copula_type in condsim_numbers:
            output = self._cond_sim()
        if self.copula_type == CopulaTypes.GUMBEL:
            output = frechet(self.rmvlog_tawn())
            output.reshape(self.n_sample, self.dim)
        if self.copula_type == CopulaTypes.ASYMMETRIC_LOGISTIC:
            number = int(2**self.dim - 1)
            dep = np.repeat(self.theta, number - self.dim)
            if (self.dim == 2) and (self.asy is None):
                self.asy = [self.psi1, self.psi2, [1-self.psi1, 1-self.psi2]]
            asy = self.mvalog_check(dep).reshape(-1)
            dep = np.concatenate([np.repeat(1, self.dim), dep], axis=None)
            output = frechet(self.rmvalog_tawn(number, dep, asy))
            output = output.reshape(self.n_sample, self.dim)
        if self.copula_type in extsim_numbers:
            output = np.exp(-1/self._ext_sim())
        return output


def frechet(var):
    """
        Probability distribution function for Frechet's law
    """
    return np.exp(-1/var)
