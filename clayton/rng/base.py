"""A multivariate copula :math:`C : [0,1]^d \mapsto [0,1]` of a d-dimensional random vector :math:`\mathbf{X}`
allows us to separate the effect of dependence from the effect of the marginal distributions such as:

.. math:: \mathbb{P}\{ X_1 \leq x_1, \dots, X_d \leq x_d \} =
                C( \mathbb{P}\{X_1 \leq x_1\}, \dots, \mathbb{P} \{X_d \leq x_d\}),

where :math:`(x_1,\dots,x_d) \in \mathbb{R}^d`. The main consequence of this identity is that 
the copula completely characterizes the stochastic dependence between the margins of :math:`\mathbf{X}`.

Structure :

- Multivariate copula (:py:class:`Multivariate`)
    - Archimedean copula (:py:mod:`clayton.rng.archimedean`)
    - Extreme value copula (:py:mod:`clayton.rng.evd`)
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

    Raises:
        ValueError:
            wrong sample size.
        ValueError:
            wrong dimension.
        ValueError:
            inv_cdf should be a list.
        ValueError:
            wrong dimension of inv_cdf.

    Returns:
        clayton.rng.base.Multivariate
    """

    @abc.abstractmethod
    def __init__(
            self,
            n_sample=1,
            dim=2,
    ):
        """Initialize Multivariate object.

        Args:
            n_sample (int, optional):
                sample size. Defaults to 1.
            dim (int, optional):
                dimension. Defaults to 2.

        Raises:
            ValueError:
                sample size is not a positive integer.
            ValueError:
                dimension is not a positive integer.
        """

        if (n_sample is None or
                (isinstance(n_sample, int) and n_sample > 0)):
            self.n_sample = n_sample
        else:
            message = "The inserted sample's size value {} \
                     should be a positive integer"
            raise ValueError(message.format(n_sample))

        if (dim is None or
                (isinstance(dim, int) and dim > 0)):
            self.dim = dim
        else:
            message = "The inserted dimension value {} \
                 should be a positive integer"
            raise ValueError(message.format(n_sample))

    def _child_method(self):
        """abstract method.
        """

    def sample_unimargin(self):
        """see the corresponding documentation in lower subclasses.
        """
        return self._child_method()

    def _generate_randomness(self):
        """Generate a bivariate sample draw identically and
        independently from a uniform over the segment [0,1].

        Returns:
            ndarray of shape (n_sample, dim).
        """
        v_1 = np.random.uniform(low=0.0, high=1.0, size=self.n_sample)
        v_2 = np.random.uniform(low=0.0, high=1.0, size=self.n_sample)
        output = np.vstack([v_1, v_2]).T
        return output

    def sample(self, inv_cdf):
        """Draws a bivariate sample the desired copula and invert it by
        a given generalized inverse of cumulative distribution function.

        Args:
            inv_cdf (list):
                list of desired margins.

        Raises:
            ValueError:
                inv_cdf should be a list.
            ValueError:
                wrong size of inv_cdf.

        Returns:
            ndarray of shape (n_sample, dim)
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

    Args:
        Multivariate (object):
            see Multivariate object.
    """
    copula_type = None
    theta_interval = None
    invalid_thetas = None
    theta = None

    @abc.abstractmethod
    def __init__(
            self,
            theta=None,
            n_sample=1,
            dim=2
    ):
        """Instantiate the Archimedean object

        Args:
            theta (float):
                parameter of the archimedean copula.
            n_sample (int):
                sample size.
            dim (int):
                dimension.

        Raises:
            ValueError:
                If there is not in :attr:`theta_interval` or
                is in :attr:`invalid_thetas`.
        """
        super().__init__(
            dim=dim,
            n_sample=n_sample
        )
        self.theta = theta
        if self.theta is not None:
            self._check_param()

    def _check_param(self):
        """Check if the parameter set by the user is correct.

        Raises:
            TypeError:
                If there is not in :attr:`theta_interval` or
                is in :attr:`invalid_thetas`.
        """
        theta_cop = [CopulaTypes.GUMBEL,
                     CopulaTypes.CLAYTON, CopulaTypes.FRANK, CopulaTypes.AMH,
                     CopulaTypes.JOE, CopulaTypes.NELSEN9, CopulaTypes.NELSEN10,
                     CopulaTypes.NELSEN11, CopulaTypes.NELSEN12, CopulaTypes.NELSEN13,
                     CopulaTypes.NELSEN14, CopulaTypes.NELSEN15, CopulaTypes.NELSEN22]
        if self.theta is not None and self.copula_type in theta_cop:
            lower, upper = self.theta_interval
            if ((self.theta < lower) | (self.theta > upper) or
                    (self.theta in self.invalid_thetas)):
                message = "The inserted theta value {} is out of limits for the \
                    given {} copula."
                raise TypeError(message.format(
                    self.theta, self.copula_type.name))

    def _child_method(self):
        """abstract method
        """

    def _generator(self, var):
        """See the documentation in archimedean.py
        """
        return self._child_method()

    def _generator_dot(self, var):
        """See the documentation in archimedean.py
        """
        return self._child_method()

    def _generator_inv(self, var):
        """See the documentation in archimedean.py
        """
        return self._child_method()

    def _rfrailty(self):
        """See the documentation in archimedean.py
        """
        return self._child_method()

    def _c(self, var):
        """Return the value of the copula taken on (u,v)
        .. math:: C(u,v) = phi^leftarrow (phi(u) + phi(v)), 0<u,v<1

        Args:
            var (ndarray of shape (, dim)):
                a real where to evaluate the copula function.

        Returns:
            real:
            the value of the copula function evaluate on var.
        """
        value_ = self._generator_inv(np.sum(self._generator(var)))
        return value_

    def _cond_sim(self):
        """Perform conditional simulation. Only useful for Archimedean copulas
        where the frailty distribution is still unknown.

        Raises:
            ValueError:
                if dim > 2.

        Returns:
            ndarray of shape (n_sample, 2):
                random numbers generated through conditional simulation.
        """

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

        Raises:
            ValueError:
                if self.theta is negative

        Returns:
            ndarray of shape (n_sample, dim):
                random numbers generated through frailty simulation.
        """

        if self.dim > 2:
            if self.theta < 0:
                message = "The inserted theta value {} is out of limits for \
                    the given {} copula. In dimension greater than 2, positive \
                        association are only allowed."
                raise ValueError(message.format(
                    self.theta, self.copula_type.name))
        output = np.zeros((self.n_sample, self.dim))
        for i in range(0, self.n_sample):
            samplegamma = np.random.gamma(1, 1, self.dim)
            samplefrailty = self._rfrailty()
            geninv = self._generator_inv(samplegamma/samplefrailty)
            output[i, :] = geninv
        return output

    def sample_unimargin(self):
        """Sample from Archimedean copula with uniform margins.
        Performs different algorithm if the frailty distribution of the
        chosen Archimedean copula is known or not.

        Returns:
            ndarray of shape (n_sample, dim):
                random numbers generated with uniform margins.
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

    Args:
        Multivariate (object):
            see Multivariate object

    """

    copula_type = None
    theta = None
    psi1 = None
    psi2 = None
    sigmat = None
    asy = None

    @abc.abstractmethod
    def _child_method(self):
        """abstract method
        """

    def _pickands(self, var):
        """See the documentation in evd.py
        """
        return self._child_method()

    def _pickandsdot(self, var, j):
        """See the documentation in evd.py
        """
        return self._child_method()

    def _mvalog_check(self, dep):
        """See the documentation in evd.py
        """
        return self._child_method()

    def _rmvlog_tawn(self):
        """See the documentation in evd.py
        """
        return self._child_method()

    def _rmvalog_tawn(self, number, alpha, asy):
        """See the documentation in evd.py
        """
        return self._child_method()

    def _rextfunc(self, index, cholesky=None):
        """See the documentation in evd.py
        """
        return self._child_method()

    def _sigma2covar(self, index):
        """See the documentation in evd.py
        """
        return self._child_method()

    def _l(self, var):
        """Return the value of the stable tail dependence function on u.
        Pickands is parametrize as A(w_0, dots, w_{d-1}) with w_0 = 1-sum_{j=1}^{d-1} w_j

        Args:
            var (list or ndarray):
                vector

        Returns:
            real:
                stable tail dependence function evaluated at var.
        """

        sumu = np.sum(var)
        vectw = var / sumu
        value_ = sumu*self._pickands(vectw)
        return value_

    def _c(self, var):
        """Return the value of the copula taken on u
        .. math:: C(u) = exp(-l(-log(u_1), dots, -log(u_d))), u in [0,1]^d.

        Args:
            var (list or ndarray):
                vector.
        Returns:
            real:
                extreme value copula evaluated at var.
        """

        log_u_ = np.log(var)
        value_ = math.exp(-self._l(-log_u_))
        return value_

    def _mu(self, var, j):
        """Return the value of the jth partial derivative of l.
        ..math:: dot{l}_j(u), u in ]0,1[^d, j in {0,dots,d-1}.

        Args:
            var (list or ndarray):
                list of float between 0 and 1.
            j (int):
                jth derivative of the stable tail dependence function.

        Returns:
            real:
                value of the jth partial derivative of l at var.
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

    def _dot_c(self, var, j):
        """Return the value of dot{C}_j taken on u.
        .. math:: dot{C}_j = C(u)/u_j * _mu_j(u), u in [0,1]^d, j in {0 , dots, d-1}.

        Args:
            var (list or ndarray):
                list of float between 0 and 1.
            j (int):
                jth derivative of the stable tail dependence function.

        Returns:
            real:
                value of dot{C}_j taken on var.
        """

        value_ = (self._c(var) / var[j]) * self._mu(-np.log(var), j)
        return value_

    def _cond_sim(self):
        """Draw a bivariate sample from an extreme value copula using conditional simulation.
        Margins are uniform.

        Raises:
            ValueError:
                if dim > 2.

        Returns:
            ndarray with shape (n_sample,2):
                random numbers generated from conditional simulation.
        """

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
        """Multivariate extreme value distribution sampling algorithm via extremal
        functions. See Dombry et al [2016], exact simulation of max-stable process
        for more details.

        Returns:
            ndarray with shape (n_sample,dim):
                random numbers generated from extremal functions.
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
                covar = self._sigma2covar(0)
                cholesky = np.linalg.cholesky(covar).T
                extfunc = self._rextfunc(0, cholesky)
            if self.copula_type in dirlog:
                extfunc = self._rextfunc(0)

            output[i, :] = extfunc / zeta

            for j in range(1, self.dim):
                zeta = np.random.exponential(1)
                if self.copula_type in matsim:
                    covar = self._sigma2covar(j)
                    cholesky = np.linalg.cholesky(covar).T

                while (1.0 / zeta > output[i, j]):
                    if self.copula_type in matsim:
                        extfunc = self._rextfunc(j, cholesky)
                    if self.copula_type in dirlog:
                        extfunc = self._rextfunc(j)
                    res = True
                    for k in range(0, j):
                        if (extfunc[k] / zeta >= output[i, k]):
                            res = False

                    if res:
                        output[i, :] = np.maximum(output[i, :], extfunc / zeta)
                    zeta += np.random.exponential(1)

        return output

    def sample_unimargin(self):
        """Sample from extreme value copula with uniform margins.
        Performs different algorithms if a fast random number generator
        is known.

        Returns:
            ndarray with shape (n_sample, dim):
                sample with uniform margins.
        """

        output = []
        condsim_numbers = [CopulaTypes.ASYMMETRIC_NEGATIVE_LOGISTIC,
                           CopulaTypes.ASYMMETRIC_MIXED_MODEL]
        extsim_numbers = [CopulaTypes.HUSLER_REISS,
                          CopulaTypes.TEV, CopulaTypes.DIRICHLET, CopulaTypes.BILOG]
        if self.copula_type in condsim_numbers:
            output = self._cond_sim()
        if self.copula_type == CopulaTypes.GUMBEL:
            output = _frechet(self._rmvlog_tawn())
            output.reshape(self.n_sample, self.dim)
        if self.copula_type == CopulaTypes.ASYMMETRIC_LOGISTIC:
            number = int(2**self.dim - 1)
            dep = np.repeat(self.theta, number - self.dim)
            if (self.dim == 2) and (self.asy is None):
                self.asy = [self.psi1, self.psi2, [1-self.psi1, 1-self.psi2]]
            asy = self._mvalog_check(dep).reshape(-1)
            dep = np.concatenate([np.repeat(1, self.dim), dep], axis=None)
            output = _frechet(self._rmvalog_tawn(number, dep, asy))
            output = output.reshape(self.n_sample, self.dim)
        if self.copula_type in extsim_numbers:
            output = np.exp(-1/self._ext_sim())
        return output


def _frechet(var):
    """Probability distribution function for _frechet's law

    Args:
        var (real):
            a real.

    Returns:
        real:
            ..math.. exp{-1/x}
    """

    return np.exp(-1/var)
