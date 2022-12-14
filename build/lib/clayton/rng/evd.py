"""Multivariate extreme value copula or, more generally, extreme value distribution
are max-stable random vector with generalized extreme value margins and we may write

.. math:: \mathbb{P}\{ \mathbf{X} \leq \mathbf{x} \} 
            = \exp\{-\Lambda(E \setminus [\mathbf{0}, \mathbf{x}])\},

where :math:`\Lambda` is a Radon measure on the cone :math:`E = [0,\infty]^d \setminus \mathbf{0}`.
This dependendence structure can be translated with the classical notion of copula, :math:`C` 
is an extreme value copula if

.. math::  C(u) = \exp\{-\ell(-\ln(u_1), \dots, -\ln(u_d))\}, 0 < u_j \leq 1,

where :math:`\ell` is the stable tail dependence function.

Structure :

- Extreme value copula (:py:class:`Extreme`) from :py:mod:`clayton.rng.base`
    - Logistic model (:py:class:`Logistic`)
    - Asymmetric logistic model (:py:class:`AsymmetricLogistic`)
    - Husler Reiss (:py:class:`HuslerReiss`)
    - Asymmetric negative logistic (:py:class:`AsyNegLog`)
    - Asymmetric mixed (:py:class:`AsyMix`)
    - t-Extreme Value (:py:class:`TEV`)
    - Bilogistic model (:py:class:`Bilog`)
"""
# pylint: disable=too-few-public-methods

import math
import abc
import numpy as np

from scipy.stats import norm
from scipy.stats import t
from .utils import rpstable, maximum_n, subsets, mvrnorm_chol_arma, rdir
from .base import CopulaTypes, Extreme


class Logistic(Extreme):
    """Class for multivariate Logistic copula model.


    Args:
        Extreme (object):
            see Extreme object.

    Raises:
        ValueError:
            if theta > 1.0 and theta < 0.0.

    Returns:
        clayton.rng.evd.Logistic:
            a Logistic object.
    """

    copula_type = CopulaTypes.GUMBEL
    theta_interval = [0, 1]
    invalid_thetas = [0]

    @abc.abstractmethod
    def __init__(
        self,
        theta=None,
        n_sample=1,
        dim=2
    ):
        """Instantiate Logistic class

        Args:
            theta (float):
                parameter between 0 and 1.
            n_sample (int):
                sample size.
            dim (int):
                dimension

        Raises:
            ValueError:
                if theta < 0.0 and theta > 1.0
        """

        super().__init__(
            n_sample=n_sample,
            dim=dim
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
        if self.theta is not None:
            lower, upper = self.theta_interval
            if ((self.theta < lower) | (self.theta > upper) or
                    (self.theta in self.invalid_thetas)):
                message = "The inserted theta value {} is out of limits for the \
                    given {} copula."
                raise ValueError(message.format(
                    self.theta, self.copula_type.name))

    def _pickands(self, var):
        """Return the value of the Pickands dependence function taken on t.
        ..math:: A(t) = (sum_{j=1}^d t_i^{1/theta})^theta, t in Delta^{d-1}.

        Args:
            var (list or ndarray):
                element of the simplex in R^{d}

        Returns:
            real:
                value of the Pickands dependence function evaluated at var
        """

        value_ = math.pow(np.sum(np.power(var, 1/self.theta)), self.theta)

        return value_

    def _pickandsdot(self, var, j):
        """Return the value of jth partial derivative of the Pickands
        dependence function taken on var

        Args:
            var (list or ndarray):
                element of the simplex in R^{d}
            j (int > 0):
                index of the partial derivative

        Returns:
            real:
                value of jth partial derivative of the Pickands
                dependence function taken on var
        """

        sumvar = np.sum(var[1:])  # sum_{j=1}^{d-1} t_j
        value_1 = (1/self.theta * math.pow(var[j], (1-self.theta)/self.theta) -
                   1/self.theta * math.pow(1-sumvar, (1-self.theta)/self.theta))
        value_2 = math.pow(self._pickands(var), (self.theta - 1)/self.theta)
        value_ = self.theta * value_1 * value_2
        return value_

    def _rmvlog_tawn(self):
        """Algorithm 2.1 of Stephenson (2002).

        Returns:
            ndarray of shape (n_sample, dim):
                Logistic dependence with Fréchet margins.
        """

        sim = np.zeros(self.n_sample * self.dim)
        for i in range(0, self.n_sample):
            rps = rpstable(self.theta)
            for j in range(0, self.dim):
                sim[i*self.dim + j] = math.exp(self.theta *
                                               (rps - math.log(np.random.exponential(size=1))))
        return sim

    def sample_unimargin(self):
        """Draws a sample from a multivariate Logistic model with uniform margins.

        Returns:
            ndarray of shape (n_sample, dim):
                Logistic dependence with uniform margins.
        """

        sim = _frechet(self._rmvlog_tawn())
        return sim.reshape(self.n_sample, self.dim)


class AsymmetricLogistic(Extreme):
    """
        Class for multivariate asymmetric logistic copula model.

    Args:
        Extreme (object):
            see Extreme object.

    Raises:
        TypeError: invalid theta.
        TypeError: asy is not a list of size 2**d -1.
        TypeError: asy does not satisfy the constraints.
        TypeError: asy does not satisfy the constraints.

    Returns:
        clayton.rng.evd.AsymmetricLogistic:
            a AsymmetricLogistic object.
    """

    copula_type = CopulaTypes.ASYMMETRIC_LOGISTIC

    @abc.abstractmethod
    def __init__(
        self,
        theta=None,
        n_sample=1,
        dim=2,
        asy=None
    ):
        """Instantiate asymmetric logistic

        Args:
            theta (float, optional):
                parameter of the copula. Defaults to None.
            n_sample (int, optional):
                sample size. Defaults to 1.
            dim (int, optional):
                dimension. Defaults to 2.
            asy (list, optional):
                asymmetry coefficients. Defaults to None.
        """

        super().__init__(
            n_sample=n_sample,
            dim=dim
        )
        self.theta = theta
        self.asy = asy
        numb = int(2**self.dim - 1)
        dep = np.repeat(self.theta, numb - self.dim)
        if self.asy is not None and self.theta is not None:
            self._mvalog_check(dep)

    def _pickands(self, var):
        """Return the value of the Pickands dependence function taken on t
        ..math:: A(t) = sum_{b in B}(sum_{j in b} (psi_{j,b} t_j)^{1/theta_b}))^{theta_b},
                        t in Delta^{d-1}

        Args:
            var (list or ndarray):
                element of the simplex in R^{d}

        Returns:
            real:
                value of the Pickands dependence function evaluated at var
        """
        numb = int(2**self.dim - 1)
        dep = np.repeat(self.theta, numb - self.dim)
        dep = np.concatenate([np.repeat(1, self.dim), dep], axis=None)
        asy = self._mvalog_check(dep)
        vecta = []
        for b in range(0, numb):
            x = np.power(asy[b, :], 1/dep[b])
            y = np.power(var, 1/dep[b])
            value = np.dot(x, y)
            vecta.append(np.power(value, dep[b]))

        return np.sum(vecta)

    def _pickandsdot(self, var, j):
        """Return the value of jth partial derivative of the Pickands
        dependence function taken on var

        Args:
            var (list or ndarray):
                element of the simplex in R^{d}
            j (int > 0):
                index of the partial derivative

        Returns:
            real:
                value of jth partial derivative of the Pickands
                dependence function taken on var
        """
        numb = int(2**self.dim - 1)
        dep = np.repeat(self.theta, numb - self.dim)
        asy = self._mvalog_check(dep)
        vectadot = []
        for b in range(0, numb):
            z = np.zeros(self.dim)
            z[0] = -np.power(var[0], (1-dep[b]) / dep[b])
            z[j] = np.power(var[j], (1-dep[b]) / dep[b])
            x = np.power(asy[b, :], 1/dep[b])
            y = np.power(t, 1/dep[b])
            value_1 = np.dot(x, z)
            value_2 = np.power(np.dot(x, y), (dep[b] - 1))
            vectadot.append(value_1 * value_2)

        return np.sum(vectadot)

    def _rmvalog_tawn(self, number, alpha, asy):
        """ Algorithm 2.2 of Stephenson (2008).

        Args:
            number (int):
                2**d-1.
            alpha (ndarray of shape (2**d-1-d)):
                concatenation of self.theta
            asy (ndarray):
                transformed asymmetry coefficients

        Returns:
            ndarray with shape (n_sample, dim):
             Asymmetric Logistic with Fréchet margins.

        """

        sim = np.zeros(self.n_sample*self.dim)
        gevsim = np.zeros(number*self.dim)
        maxsim = np.zeros(number)
        for i in range(0, self.n_sample):
            for j in range(0, number):
                if alpha[j] != 1:
                    rps = rpstable(alpha[j])
                else:
                    rps = 0
                for k in range(0, self.dim):
                    if asy[j*self.dim+k] != 0:
                        gevsim[j*self.dim+k] = asy[j*self.dim+k] * \
                            math.exp(
                                alpha[j] * (rps - math.log(np.random.exponential(size=1))))

            for j in range(0, self.dim):
                for k in range(0, number):
                    maxsim[k] = gevsim[k*self.dim+j]

                sim[i*self.dim+j] = maximum_n(number, maxsim)

        return sim

    def _mvalog_check(self, dep):
        """Check value of theta and asy

        Args:
            dep (list):
                concatenation 2**d-1-d of self.theta.

        Raises:
            TypeError: theta incorrect.
            TypeError: asy is not a list with wrong size.
            TypeError: asy does not satisfy constraints.
            TypeError: asy does not satisfy constraints.

        Returns:
            asy (ndarray):
                transformed asy.
        """
        if np.any((dep < 0.0) | (dep > 1.0)):
            raise TypeError('invalid argument for theta')
        numb = 2 ** self.dim - 1
        if (not isinstance(self.asy, list) or len(self.asy) != numb):
            raise TypeError('asy should be a list of length', numb)

        def tasy(theta, sub):
            trans = np.zeros([numb, self.dim])
            for i in range(0, numb):
                j = sub[i]
                trans[i, j] = theta[i]
            return trans

        sub = subsets(self.dim)
        asy = tasy(self.asy, sub)
        sumy = np.sum(asy, axis=0)
        indices = [index for index in range(len(dep)) if dep[index] == 1.0]
        if (sumy != 1).any():
            raise TypeError(
                "asy does not satisfy the appropriate constraints, sum should be equal to 1")
        for index in indices:
            if np.sum(dep[index]) > 0 and (index >= self.dim):
                raise TypeError(
                    "asy does not satisfy the appropriate constrains")
        return asy


class HuslerReiss(Extreme):
    """Class for Husler Reiss copula model.

    Args:
        Extreme (object):
            Extreme object

    Raises:
        ValueError: if not an array
        ValueError: if not a squared matrix
        ValueError: if not CNSD

    Returns:
        _type_: _description_
    """

    copula_type = CopulaTypes.HUSLER_REISS

    @abc.abstractmethod
    def __init__(
        self,
        sigmat=None,
        n_sample=1,
        dim=2
    ):
        """Instantiate HuslerReiss copula model

        Args:
            sigmat (ndarray, optional):
                ndarray with shape (dim,dim). Defaults to None.
            n_sample (int, optional):
                sample size. Defaults to 1.
            dim (int, optional):
                dimension. Defaults to 2.
        """

        super().__init__(
            n_sample=n_sample,
            dim=dim
        )
        self.sigmat = sigmat
        if self.sigmat is not None:
            self._check_cnsd()

    def _pickands(self, var):
        """Return the generator function.
        .. math:: A(t) = (1-t) * Phi(theta + frac{1}{2theta}logfrac{1-t}{t})
                        + t * Phi(theta + frac{1}{2theta}logfrac{t}{1-t}),
                        0 < t < 1.
        Args:
            var (list or ndarray):
                element of the simplex in R^{d}

        Returns:
            real:
                value of the Pickands dependence function evaluated at var
        """
        value_1 = var[0] * norm.cdf(self.theta + 1/(2*self.theta)
                                    * math.log(var[0]/var[1]))  # var[0] = (1-t), var[1] = t
        value_2 = var[1] * norm.cdf(self.theta + 1 /
                                    (2*self.theta)*math.log(var[1]/var[1]))
        return value_1 + value_2

    def _pickandsdot(self, var, j):
        """Return the value of jth partial derivative of the Pickands
        dependence function taken on var

        Args:
            var (list or ndarray):
                element of the simplex in R^{d}
            j (int > 0):
                index of the partial derivative

        Returns:
            real:
                value of jth partial derivative of the Pickands
                dependence function taken on var
        """
        value_1 = norm.cdf(self.theta + 1 / (2*self.theta)
                           * math.log(var[0]/var[1]))
        value_2 = (1/var[1]) * norm.pdf(self.theta + 1 /
                                        (2*self.theta) * math.log(var[0]/var[1]))
        value_3 = norm.cdf(self.theta + 1/(2*self.theta)
                           * math.log(var[1]/var[0]))
        value_4 = (1/var[0]) * norm.pdf(self.theta + 1 /
                                        (2*self.theta) * math.log(var[1]/var[0]))
        return - value_1 - value_2 + value_3 + value_4

    def _check_cnsd(self, tol=1e-08):
        """Is the matrix conditionally negative semi-definite?
        Function adapted from '.is.CNSD' in the mev package

        Args:
            sigmat (_type_): symmetric matrix
            tol (_type_, optional): tolerance value. Defaults to 1e-08.

        Raises:
            ValueError: if not an array
            ValueError: if not a squared matrix
            ValueError: if not CNSD
        """
        if isinstance(self.sigmat, np.ndarray):
            if not self.sigmat.shape[0] == self.sigmat.shape[1]:
                message = "{} should be a squared matrix"
                raise ValueError(message.format(self.sigmat))
        else:
            message = "{} should be an array"
            raise ValueError(message.format(self.sigmat))
        nrow = self.sigmat.shape[0]
        diagn = np.zeros((nrow, nrow), int)  # Create matrix with only 0
        np.fill_diagonal(diagn, 1)  # fill diagonal with 1
        if nrow > 2:
            diagn_minus = np.delete(diagn, 0, axis=1)
            np.fill_diagonal(diagn_minus, -1)
            diagn = np.concatenate(
                [np.eye(nrow)[0].reshape(nrow, 1), diagn_minus], axis=1)
        elif nrow == 2:
            diagn[0, 1] = -1

        xhat = diagn @ self.sigmat @ diagn.T
        eigs = np.linalg.eig(
            np.delete(np.delete(xhat, nrow-1, 0), nrow-1, 1))[0]
        if eigs[0] > tol:
            message = "{} should be conditionally negative semi-definite"
            raise ValueError(message.format(self.sigmat))

    def _sigma2covar(self, index):
        """Transform positive definite covariance matrix to a
        conditionally negative definite matrix (see Engelke and Hitz, 2020 Appendix B).

        Args:
            index (int):
                index of the location. An integer in {0, ..., d-1}

        Returns:
            ndarray of shape (dim,dim):
                a matrix positive definite
        """

        covar = 0.5 * (np.repeat(self.sigmat[:, index], self.sigmat.shape[0]).reshape(
            self.sigmat.shape[0],
            self.sigmat.shape[0]) +
            np.repeat(self.sigmat[index, :], self.sigmat.shape[1]).reshape(
                self.sigmat.shape[0], self.sigmat.shape[0], order='F') - self.sigmat)
        covar = np.delete(covar, index, axis=0)
        covar = np.delete(covar, index, axis=1)
        return covar

    def _rextfunc(self, index, cholesky):
        """ Generate from extremal Husler-Reiss distribution Y follows P_x, where
        P_x is probability of extremal function

        Args:
            index (int):
                index of the location. An integer in {0, ..., d-1}.
            cholesky (ndarray):
                the Cholesky root of sigmat

        Returns:
            ndarray with shape (n_sample, dim):
                sample from an HuslerReiss copula with Fréchet margins.
        """

        gamma = self.sigmat[:, index] / 2
        gamma = np.delete(gamma, index)
        normalsamp = mvrnorm_chol_arma(1, gamma, cholesky)

        indexentry = 0
        normalsamp = np.insert(normalsamp, index, indexentry)
        gamma = np.insert(gamma, index, indexentry)
        samp = np.exp(normalsamp)
        samp[index] = 1.0
        return samp


class AsyNegLog(Extreme):
    """Class for asymmetric negative logistic copula model.

    Args:
        Extreme (object):
            Extreme object.

    Raises:
        TypeError: invalid theta.
        TypeError: invalid psi1 or psi2.

    Returns:
        clayton.rng.evd.asyneglog:
            a AsyNegLog object.
    """

    copula_type = CopulaTypes.ASYMMETRIC_NEGATIVE_LOGISTIC
    theta_interval = [1, float('inf')]
    invalid_thetas = []

    @abc.abstractmethod
    def __init__(
        self,
        theta=None,
        psi1=None,
        psi2=None,
        n_sample=1,
        dim=2
    ):
        """Instantiate the asymmetric negatic logistic copula model.

        Args:
            theta (float, optional):
                parameter of the copula. Defaults to None.
            psi1 (float, optional):
                first coefficient of asymmetry. Defaults to None.
            psi2 (float, optional):
                second coefficient of asymmetry. Defaults to None.
            n_sample (int, optional):
                sample size. Defaults to 1.
            dim (int, optional):
                dimension. Defaults to 2.
        """
        super().__init__(
            n_sample=n_sample,
            dim=dim
        )
        self.theta = theta
        self.psi1, self.psi2 = psi1, psi2
        if (self.psi1 is not None and
            self.psi2 is not None and
                self.theta is not None):
            self._check_param()

    def _check_param(self):
        """Check parameters of the asymmetric negative logistic copula model.

        Raises:
            TypeError: theta is out of bounds.
            TypeError: psi1 or psi2 are out of bounds.
        """
        lower, upper = self.theta_interval
        if ((self.theta < lower) | (self.theta > upper) or
                (self.theta in self.invalid_thetas)):
            message = "The inserted theta value {} is out of limits for \
                       the given {} copula."
            raise TypeError(message.format(
                self.theta, self.copula_type.name))
        if ((self.psi1 < 0.0) | (self.psi1 > 1.0) or
                (self.psi2 < 0.0) | (self.psi2 > 1.0)):
            message = "The interseted asymmetric coefficients {} and \
                        {} should be between 0 and 1 for the given {} copula."
            raise TypeError(message.format(
                self.psi1, self.psi2, self.copula_type.name))

    def _pickands(self, var):
        """Return the Pickands dependence function.
        .. math:: A(t) = 1-[(psi_1(1-t))^{-theta} + (psi_2t)^{-theta}]^frac{1}{theta},
                    0 < t < 1.

        Args:
            var (list or ndarray):
                element of the simplex in R^{d}

        Returns:
            real:
                value of the Pickands dependence function evaluated at var
        """
        value_ = 1-math.pow(math.pow(self.psi1*var[0], -1*self.theta) + math.pow(
            self.psi2*var[1], -1*self.theta), -1*1/self.theta)
        return value_

    def _pickandsdot(self, var, j=0):
        """Return the value of jth partial derivative of the Pickands
        dependence function taken on var

        Args:
            var (list or ndarray):
                element of the simplex in R^{d}
            j (int > 0):
                index of the partial derivative

        Returns:
            real:
                value of jth partial derivative of the Pickands
                dependence function taken on var
        """
        value_1 = 1/(var[0]*math.pow(self.psi1*var[0], self.theta)) - \
            1/(var[1]*math.pow(self.psi2*var[1], self.theta))
        value_2 = math.pow(self.psi2*var[1], -1*self.theta) + \
            math.pow(self.psi1*var[0], -1*self.theta)
        return value_1*math.pow(value_2, -1/self.theta-1)


class AsyMix(Extreme):
    """Class for asymmetric mixed model.

    Args:
        Extreme (object):
            Extreme object.

    Raises:
        ValueError:
            theta and psi1 does not verify contraints.

    Returns:
        _type_:
            clayton.rng.evd.AsyMix
    """

    copula_type = CopulaTypes.ASYMMETRIC_MIXED_MODEL
    theta_interval = [0, float('inf')]
    invalid_thetas = []

    @abc.abstractmethod
    def __init__(
        self,
        theta=None,
        psi1=None,
        n_sample=1,
        dim=2
    ):
        """Instantiate AsyMix copula model.

        Args:
            theta (float, optional):
                parameter of the copula. Defaults to None.
            psi1 (float, optional):
                parameter of asymmetry. Defaults to None.
            n_sample (int, optional):
                sample size. Defaults to 1.
            dim (int, optional):
                dimension. Defaults to 2.
        """
        super().__init__(
            n_sample=n_sample,
            dim=dim
        )

        self.theta = theta
        self.psi1 = psi1
        if self.theta is not None and self.psi1 is not None:
            self._check_param()

    def _check_param(self):
        """
            Validate the parameters inserted.
            This method is used to assert if the parameters are
            in the valid range for the model.

            Raises :
                ValueError : If theta or psi_1 does not satisfy the constraints.
        """

        if (not self.theta >= 0) or (not self.theta + 3*self.psi1 >= 0) or \
                (not self.theta + self.psi1 <= 1) or (not self.theta + 2*self.psi1 <= 1):
            message = "Parameters inserted {}, {} does not satisfy \
               the inequalities for the given {} copula"
            raise ValueError(message.format(
                self.theta, self.psi1, self.copula_type.name))

    def _pickands(self, var):
        """Return the Pickands dependence function.
        .. math:: A(t) = 1-(theta+psi_1)*t + theta*t^2 + psi_1 * t^3,  0 < t < 1.

        Args:
            var (list or ndarray):
                element of the simplex in R^{d}

        Returns:
            real:
                value of the Pickands dependence function evaluated at var
        """
        value_ = 1-(self.theta + self.psi1) * \
            var[1] + self.theta * \
            math.pow(var[1], 2) + self.psi1*math.pow(var[1], 3)
        return value_

    def _pickandsdot(self, var, j=0):
        """Return the value of jth partial derivative of the Pickands
        dependence function taken on var

        Args:
            var (list or ndarray):
                element of the simplex in R^{d}
            j (int > 0):
                index of the partial derivative

        Returns:
            real:
                value of jth partial derivative of the Pickands
                dependence function taken on var
        """
        value_ = -(self.theta+self.psi1) + 2*self.theta * \
            var[1]+3*self.psi1*math.pow(var[1], 2)
        return value_


class TEV(Extreme):
    """Class for t extreme value model.

    Args:
        Extreme (Extreme):
            Extreme object.

    Raises:
        ValueError:
            psi1 is not positive.
        ValueError:
            sigmat is not a squared matrix.

    Returns:
        clayton.rng.evd.TEV:
            a TEV object.
    """

    copula_type = CopulaTypes.TEV
    theta_interval = [-1, 1]
    invalid_thetas = []

    @abc.abstractmethod
    def __init__(
        self,
        sigmat=None,
        psi1=None,
        n_sample=1,
        dim=2
    ):
        """Instantiate TEV copula model.

        Args:
            sigmat (ndarray, optional):
                ndarray of shape (dim,dim). Defaults to None.
            psi1 (float, optional):
                positive float. Defaults to None.
            n_sample (int, optional):
                sample size. Defaults to 1.
            dim (int, optional):
                dimension. Defaults to 2.
        """
        super().__init__(
            n_sample=n_sample,
            dim=dim
        )

        self.sigmat = sigmat
        self.psi1 = psi1

        if self.sigmat is not None and self.psi1 is not None:
            self._check_param()

    def _check_param(self):
        """Check sigmat and psi1.

        Raises:
            ValueError: psi1 should be positive.
            ValueError: sigmat should be a squared matrix.
        """
        if self.psi1 <= 0:
            message = "The parameter {} should be a positive quantity for \
                the {} copula."
            raise ValueError(message.format(self.psi1, self.copula_type))
        if isinstance(self.sigmat, np.ndarray):
            if (not self.sigmat.shape[0] == self.sigmat.shape[1] or
                    not np.allclose(self.sigmat, self.sigmat.T)):
                message = "{} should be a squared matrix"
                raise ValueError(message.format(self.sigmat))

    def _ztev(self, var):
        """Intermediate quantity

        Args:
            var (list or ndarray)

        Returns:
            float
        """
        value_ = math.pow((1+self.psi1), 1/2)*(math.pow(var/(1-var), 1/self.psi1) -
                                               self.theta)*math.pow(1-math.pow(self.theta, 2), -1/2)
        return value_

    def _pickands(self, var):
        """Return the Pickands dependence function.
        .. math:: A(w) = wt_{chi+1}(z_w)+(1-w)t_{chi+1}(z_{1-w})  0 < w < 1.
        .. math:: z_w  = (1+chi)^frac{1}{2}[(w/(1-w))^frac{1}{chi}
                         - rho](1-rho^2)^frac{-1}{2}.

        Args:
            var (list or ndarray):
                element of the simplex in R^{d}

        Returns:
            real:
                value of the Pickands dependence function evaluated at var
        """
        value_ = var[1]*t.cdf(self._ztev(var[1]), df=self.psi1 + 1) + \
            (1-var[1])*t.cdf(self._ztev(var[0]), df=self.psi1 + 1)
        return value_

    def _pickandsdot(self, var, j=0):
        """Return the value of jth partial derivative of the Pickands
        dependence function taken on var

        Args:
            var (list or ndarray):
                element of the simplex in R^{d}
            j (int > 0):
                index of the partial derivative

        Returns:
            real:
                value of jth partial derivative of the Pickands
                dependence function taken on var
        """
        value_1 = t.cdf(self._ztev(var[1]), df=self.psi1 + 1)
        value_2 = (1/var[0]) * t.pdf(self._ztev(var[1]), df=self.psi1+1) * \
            math.pow((1+self.psi1), 1/2) * \
            math.pow(1-math.pow(self.theta, 2), -1/2) * \
            math.pow(var[1]/var[0], 1/self.psi1)
        value_3 = t.cdf(self._ztev(var[0]), df=self.psi1 + 1)
        value_4 = (1/var[1]) * t.pdf(self._ztev(var[0]), df=self.psi1 + 1) * \
            math.pow((1+self.psi1), 1/2) * \
            math.pow(1-math.pow(self.theta, 2), -1/2) * \
            math.pow(var[0]/var[1], 1/self.psi1)
        return value_1 + value_2 - value_3 - value_4

    def _sigma2covar(self, index):
        """Transform positive definite covariance matrix to a
        conditionally negative definite matrix (see Engelke and Hitz, 2020 Appendix B).

        Args:
            index (int):
                index of the location. An integer in {0, ..., d-1}

        Returns:
            ndarray of shape (dim,dim):
                a matrix positive definite
        """
        covar = (self.sigmat - self.sigmat[:, index].reshape(self.sigmat.shape[0], 1)
                 @ self.sigmat[index, :].reshape(1, self.sigmat.shape[1])) / (self.psi1 + 1.0)
        covar = np.delete(covar, index, axis=0)
        covar = np.delete(covar, index, axis=1)
        return covar

    def _rextfunc(self, index, cholesky):
        """ Generate from extremal Student-t probability of extremal function

        Args:
            index (int):
                index of the location. An integer in {0, ..., d-1}.
            cholesky (ndarray):
                the Cholesky root of sigmat

        Returns:
            ndarray with shape (n_sample, dim):
                sample from an HuslerReiss copula with Fréchet margins.

        https://github.com/lbelzile/mev/blob/main/src/sampling.cpp
        """
        zeromean = np.zeros(self.sigmat.shape[1]-1)
        normalsamp = mvrnorm_chol_arma(1, zeromean, cholesky)
        indexentry = 0
        normalsamp = np.insert(normalsamp, index, indexentry)
        chisq = np.random.chisquare(self.psi1 + 1.0, size=1)
        studsamp = np.exp(0.5 * (np.log(self.psi1 + 1.0) - np.log(chisq))) * \
            normalsamp + np.squeeze(np.asarray(self.sigmat[:, index]))
        samp = np.power(np.maximum(studsamp, 0), self.psi1)
        samp[index] = 1.0
        return samp


class Dirichlet(Extreme):
    """Class for Dirichlet mixture models.

    Args:
        Extreme (object):
            Extreme object.

    Returns:
        clayton.rng.evd.Dirichlet:
    """

    copula_type = CopulaTypes.DIRICHLET

    @abc.abstractmethod
    def __init__(
        self,
        sigmat=None,
        theta=None,
        n_sample=1,
        dim=2
    ):
        """Instantiate Dirichlet mixture model.

        Args:
            sigmat (ndarray, optional):
                ndarray of shape (m,dim) where m is the number in the mixture.
                Defaults to None.
            theta (float, optional):
                list of positive float, sum to 1. Defaults to None.
            n_sample (int, optional):
                sample size. Defaults to 1.
            dim (int, optional):
                dimension. Defaults to 2.
        """
        super().__init__(
            n_sample=n_sample,
            dim=dim
        )

        self.sigmat = sigmat
        self.theta = theta

        if self.sigmat is not None and self.theta is not None:
            self._check_param()

    def _check_param(self):
        """Check parameter theta

        Raises:
            ValueError:
                invalid argument for theta
        """
        if np.sum(self.theta) != 1.0:
            raise ValueError('sum of theta should be equal to one')
        if np.any((self.theta < 0.0) | (self.theta > 1.0)):
            raise ValueError('each component of theta should be between 0 and 1')
        if ((self.sigmat.shape[0] != len(self.theta)) or
                (self.sigmat.shape[1] != self.dim) or
                np.any(self.sigmat < 0.0)):
            raise ValueError('invalid argument for sigmat')

    def _rextfunc(self, index):
        """ Generate from extremal Dirichlet Y follows P_x, where
        P_{x} is the probability of extremal functions from a Dirichlet mixture

        Args:
            index (int):
                index of the location.

        Returns:
            ndarray of dim (d)
        """
        int_seq = np.arange(self.dim)
        # Probability weights
        weights = np.zeros(len(self.theta))
        for k in range(0, len(self.theta)):
            weights[k] = len(self.theta) * self.theta[k] * \
                self.sigmat[index, k] / sum(self.sigmat[:, k])

        m = np.random.choice(int_seq, 1, False, weights)[0]

        sample = np.zeros(self.dim)
        gzero = np.random.gamma(self.sigmat[index, m] + 1.0, 1.0, size=1)
        for j in range(0, self.dim):
            sample[j] = np.random.gamma(self.sigmat[j, m], 1.0, size=1) / gzero
        sample[index] = 1.0
        return sample


class Bilog(Extreme):
    """ Class for bilogistic distribution model Smith (1990). """

    copula_type = CopulaTypes.BILOG

    @abc.abstractmethod
    def __init__(
            self,
            theta=None,
            n_sample=1,
            dim=2
    ):
        """Instantiate Bilog model

        Args:
            theta (ndarray, optional):
                parameter of the model. Defaults to None.
            n_sample (int, optional):
                sample size. Defaults to 1.
            dim (int, optional):
                dimension. Defaults to 2.
        """
        super().__init__(
            n_sample=n_sample,
            dim=dim
        )
        self.theta = theta
        if self.theta is not None:
            self._check_param()

    def _check_param(self):
        """Check parameter theta

        Raises:
            ValueError:
                invalid argument for theta
        """
        if np.any((self.theta < 0.0) | (self.theta > 1.0)):
            raise ValueError('invalid argument for theta')

    def _pickands(self, var):
        raise NotImplementedError

    def _pickandsdot(self, var, j=0):
        raise NotImplementedError

    def _rextfunc(self, index, normalize=True):
        """Random variate generation for Dirichlet distribution on S_d
        A function to sample Dirichlet random variables, based on the representation
        as ratios of Gamma.

        Args:
            index (int)
            normalize (bool, optional):
                If code{False}, the function returns Gamma variates with parameter
                code{theta}. Defaults to True.

        Returns:
            ndarray with shape (n_sample, dim):
                Bilogistic random numbers with Fréchet margins.
        """
        alpha_star = np.ones(self.dim)
        sample = np.zeros(self.dim)
        alpha_star[index] = 1.0 - self.theta[index]
        sample = rdir(1, alpha_star, normalize)[0, :]
        for i in range(0, self.dim):
            sample[i] = np.exp(-self.theta[i] * np.log(sample[i]) +
                               math.lgamma(self.dim-self.theta[i]) - math.lgamma(1-self.theta[i]))
        sample = sample / sample[index]
        return sample


def _frechet(var):
    """
        Probability distribution function for _frechet's law
    """
    return np.exp(-1/var)

# aide doc https://developer.lsst.io/v/u-ktl-debug-fix/docs/rst_styleguide.html
