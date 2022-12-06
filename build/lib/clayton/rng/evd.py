"""Multivariate extreme value copula module contains methods for sampling from a multivariate
extreme value copula and to compute the asymptotic variance of the w-madogram under missing or
complete data.

Multivariate extreme value copulas are characterized by their stable tail dependence function
which the restriction to the unit simplex gives the Pickands dependence function. The copula
function

..math:: C(u) = exp{-l(-log(u_1), dots, -log(u_d))},  0 < u_j <= 1,

is a multivariate extreme value copula. To sample from a multivariate extreme value copula, we
implement the Algoritm 2.1 and 2.2 from Stephenson (2002).

Structure :

- Extreme value copula (:py:class:`Extreme`) from copy.multivariate.base.py
    - Logistic model (:py:class:`Logistic`)
    - Asymmetric logistic model (:py:class:`Asymmetric_logistic`)
"""
# pylint: disable=too-few-public-methods

import math
import numpy as np
import numpy.matlib

from scipy.stats import norm
from scipy.stats import t
from .utils import rpstable, maximum_n, subsets, mvrnorm_chol_arma, rdir
from .base import CopulaTypes, Extreme


class Logistic(Extreme):
    """
        Class for multivariate Logistic copula model.
    """

    copula_type = CopulaTypes.GUMBEL
    theta_interval = [0, 1]
    invalid_thetas = [0]

    def _pickands(self, var):
        """Return the value of the Pickands dependence function taken on t.
        ..math:: A(t) = (sum_{j=1}^d t_i^{1/theta})^theta, t in Delta^{d-1}.

        Inputs
        ------
            t (list[float]) : list of elements of the simplex in R^{d}
        """
        value_ = math.pow(np.sum(np.power(var, 1/self.theta)), self.theta)

        return value_

    def _pickandsdot(self, var, j):
        """Return the value of jth partial derivative of the Pickands dependence function taken on t

        Inputs
        ------
            t(list[float]) : list of elements of the simplex in R^{d}
                         j : index of the partial derivative >= 1
        """
        sumvar = np.sum(var[1:])  # sum_{j=1}^{d-1} t_j
        value_1 = (1/self.theta * math.pow(var[j], (1-self.theta)/self.theta) -
                   1/self.theta * math.pow(1-sumvar, (1-self.theta)/self.theta))
        value_2 = math.pow(self._pickands(var), (self.theta - 1)/self.theta)
        value_ = self.theta * value_1 * value_2
        return value_

    def rmvlog_tawn(self):
        """ Algorithm 2.1 of Stephenson (2002).
        """
        sim = np.zeros(self.n_sample * self.dim)
        for i in range(0, self.n_sample):
            rps = rpstable(self.theta)
            for j in range(0, self.dim):
                sim[i*self.dim + j] = math.exp(self.theta *
                                               (rps - math.log(np.random.exponential(size=1))))
        return sim

    def sample_unimargin(self):
        """Draws a sample from a multivariate Logistic model.

        Output
        ------
        sim (np.array([float])) : dataset of shape n_sample x d
        """
        sim = self.frechet(self.rmvlog_tawn())
        return sim.reshape(self.n_sample, self.dim)


class AsymmetricLogistic(Extreme):
    """
        Class for multivariate asymmetric logistic copula model
    """

    copula_type = CopulaTypes.ASYMMETRIC_LOGISTIC

    def _pickands(self, var):
        """Return the value of the Pickands dependence function taken on t
        ..math:: A(t) = sum_{b in B}(sum_{j in b} (psi_{j,b} t_j)^{1/theta_b}))^{theta_b},
                        t in Delta^{d-1}

        Inputs
        ------
            t (list[float]) : list of elements of the simplex in R^{d}
        """
        numb = int(2**self.dim - 1)
        dep = np.repeat(self.theta, numb - self.dim)
        dep = np.concatenate([np.repeat(1, self.dim), dep], axis=None)
        asy = self.mvalog_check(dep)
        vecta = []
        for b in range(0, numb):
            x = np.power(asy[b, :], 1/dep[b])
            y = np.power(var, 1/dep[b])
            value = np.dot(x, y)
            vecta.append(np.power(value, dep[b]))

        return np.sum(vecta)

    def _pickandsdot(self, var, j):
        """Return the value of jth partial derivative of the Pickands dependence function taken on t

        Inputs
        ------
            t(list[float]) : list of elements of the simplex in R^{d-1}
                         j : index of the partial derivative >= 1
        """
        numb = int(2**self.dim - 1)
        dep = np.repeat(self.theta, numb - self.dim)
        dep = np.concatenate([np.repeat(1, self.dim), dep], axis=None)
        asy = self.mvalog_check(dep)
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

    def rmvalog_tawn(self, number, alpha, asy):
        """ Algorithm 2.2 of Stephenson (2008). """
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

    def mvalog_check(self, dep):
        """Check and transform the dependence parameter.
        """
        if (dep.any() <= 0 or dep.any() > 1.0):
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
        if sumy.any() != 1.0:
            raise TypeError(
                "asy does not satisfy the appropriate constraints, sum")
        for index in indices:
            if np.sum(dep[index]) > 0 and (index >= self.dim):
                raise TypeError(
                    "asy does not satisfy the appropriate constrains")
        return asy


class HuslerReiss(Extreme):
    """Class for Hussler_Reiss copula model"""

    copula_type = CopulaTypes.HUSLER_REISS
    theta_interval = [0, float('inf')]
    invalid_thetas = []

    def _pickands(self, var):
        """Return the generator function.
        .. math:: A(t) = (1-t) * Phi(theta + frac{1}{2theta}logfrac{1-t}{t})
                        + t * Phi(theta + frac{1}{2theta}logfrac{t}{1-t}),
                        0 < t < 1

        Inputs
        ------
            u (list[1-float,float]) : points of the simplexe we evaluate the pickands.
        """
        value_1 = var[0] * norm.cdf(self.theta + 1/(2*self.theta)
                                    * math.log(var[0]/var[1]))  # var[0] = (1-t), var[1] = t
        value_2 = var[1] * norm.cdf(self.theta + 1 /
                                    (2*self.theta)*math.log(var[1]/var[1]))
        return value_1 + value_2

    def _pickandsdot(self, var, j):
        """Return the derivative

        Inputs
        ------
            u (list[1-float,float]) : points of the simplexe we evaluate the pickands.
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

    def sigma2covar(self, index):
        """ Transform positive definite covariance matrix to a
        conditionally negative definite matrix (see Engelke and Hitz, 2020 Appendix B).

        Input
        -----
            index : index of the location. An integer in {0, ..., d-1}

        """
        covar = 0.5 * (np.matlib.repmat(self.sigmat[:, index], 1, self.sigmat.shape[0]) +
                       np.matlib.repmat(self.sigmat[index, :], self.sigmat.shape[1], 1) -
                       self.sigmat)
        covar = np.delete(covar, index, axis=0)
        covar = np.delete(covar, index, axis=1)
        return covar

    def rextfunc(self, index, cholesky):
        """ Generate from extremal Husler-Reiss distribution Y follows P_x, where
        P_x is probability of extremal function

        Input
        -----
            index   : index of the location. An integer in {0, ..., d-1}
            sigmat  : a covariance matrix formed from the symmetric square
                      matrix of coefficients lambda^2
            cholesky: the Cholesky root of sigmat

        Output
        ------
            d-vector from P_x

        https://github.com/lbelzile/mev/blob/main/src/sampling.cpp
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
    """Class for asymmetric negative logistic copula model."""

    copula_type = CopulaTypes.ASYMMETRIC_NEGATIVE_LOGISTIC
    theta_interval = [1, float('inf')]
    invalid_thetas = []

    def _pickands(self, var):
        """Return the Pickands dependence function.
        .. math:: A(t) = 1-[(psi_1(1-t))^{-theta} + (psi_2t)^{-theta}]^frac{1}{theta},  0 < t < 1.

         Inputs
        ------
            u (list[1-float,float]) : points of the simplexe we evaluate the pickands.
        """
        value_ = 1-math.pow(math.pow(self.psi1*var[0], -self.theta) + math.pow(
            self.psi2*var[1], -self.theta), -1/self.theta)
        return value_

    def _pickandsdot(self, var, j=0):
        """Return the derivative of the Pickands dependence function.
        """
        value_1 = 1/(var[0]*math.pow(self.psi1*var[0], self.theta)) - \
            1/(var[1]*math.pow(self.psi2*var[1], self.theta))
        value_2 = math.pow(self.psi2*var[1], -self.theta) + \
            math.pow(self.psi1*var[0], -self.theta)
        return value_1*math.pow(value_2, -1/self.theta-1)


class AsyMix(Extreme):
    """Class for asymmetric mixed model"""

    copula_type = CopulaTypes.ASYMMETRIC_MIXED_MODEL
    theta_interval = [0, float('inf')]
    invalid_thetas = []

    def _pickands(self, var):
        """Return the Pickands dependence function.
        .. math:: A(t) = 1-(theta+psi_1)*t + theta*t^2 + psi_1 * t^3,  0 < t < 1.

         Inputs
        ------
            u (list[1-float,float]) : points of the simplexe we evaluate the pickands.
        """
        value_ = 1-(self.theta + self.psi1) * \
            var[1] + self.theta * \
            math.pow(var[1], 2) + self.psi1*math.pow(var[1], 3)
        return value_

    def _pickandsdot(self, var, j=0):
        """Return the derivative of the Pickands dependence function.
        """
        value_ = -(self.theta+self.psi1) + 2*self.theta * \
            var[1]+3*self.psi1*math.pow(var[1], 2)
        return value_

    def check_parameters(self):
        """
            Validate the parameters inserted.

            This method is used to assert if the parameters are in the valid range for the model.

            Raises :
                ValueError : If theta or psi_1 does not satisfy the constraints.
        """

        if (not self.theta >= 0) or (not self.theta + 3*self.psi1 >= 0) or \
                (not self.theta + self.psi1 <= 1) or (self.theta + 2*self.psi1 <= 1):
            message = 'Parameters inserted {}, {} does not satisfy the inequalities for the given {} copula'
            raise ValueError(message.format(
                self.theta, self.psi1, self.copulaTypes.name))


class TEV(Extreme):
    """Class for t extreme value model"""

    copula_type = CopulaTypes.TEV
    theta_interval = [-1, 1]
    invalid_thetas = []

    def ztev(self, var):
        """Intermediate quantity to compute the value of the TEV's pickands.
        """
        value_ = math.pow((1+self.psi1), 1/2)*(math.pow(var/(1-var), 1/self.psi1) -
                                               self.theta)*math.pow(1-math.pow(self.theta, 2), -1/2)
        return value_

    def _pickands(self, var):
        """Return the Pickands dependence function.
        .. math:: A(w) = wt_{chi+1}(z_w)+(1-w)t_{chi+1}(z_{1-w})  0 < w < 1.
        .. math:: z_w  = (1+chi)^frac{1}{2}[(w/(1-w))^frac{1}{chi} - \rho](1-\rho^2)^frac{-1}{2}.

         Inputs
        ------
            u (list[1-float,float]) : points of the simplexe we evaluate the pickands.
        """
        value_ = var[1]*t.cdf(self.ztev(var[1]), df=self.psi1 + 1) + \
            (1-var[1])*t.cdf(self.ztev(var[0]), df=self.psi1 + 1)
        return value_

    def _pickandsdot(self, var, j=0):
        """Return the derivative of the Pickands dependence function.
        """
        value_1 = t.cdf(self.ztev(var[1]), df=self.psi1 + 1)
        value_2 = (1/var[0]) * t.pdf(self.ztev(var[1]), df=self.psi1+1) * \
            math.pow((1+self.psi1), 1/2) * \
            math.pow(1-math.pow(self.theta, 2), -1/2) * \
            math.pow(var[1]/var[0], 1/self.psi1)
        value_3 = t.cdf(self.ztev(var[0]), df=self.psi1 + 1)
        value_4 = (1/var[1]) * t.pdf(self.ztev(var[0]), df=self.psi1 + 1) * \
            math.pow((1+self.psi1), 1/2) * \
            math.pow(1-math.pow(self.theta, 2), -1/2) * \
            math.pow(var[0]/var[1], 1/self.psi1)
        return value_1 + value_2 - value_3 - value_4

    def sigma2covar(self, index):
        """ Operation on the covariance matrix to sample from the extremal function.
        Input
        -----
            index : index of the location. An integer in {0, ..., d-1}

        """
        covar = (self.sigmat - np.matrix(self.sigmat[:, index])
                 @ np.matrix(self.sigmat[index, :])) / (self.psi1 + 1.0)
        covar = np.delete(covar, index, axis=0)
        covar = np.delete(covar, index, axis=1)
        return covar

    def rextfunc(self, index, cholesky):
        """ Generate from extremal Student-t probability of extremal function

        Input
        -----
            index   : index of the location. An integer in {0, ..., d-1}
            sigmat  : a positive semi-definite correlation matrix
            cholesky: Cholesky root of transformed correlation matrix
            alpha   : the alpha parameter. Corresponds to degrees of freedom - 1

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
    """ Class for Dirichlet mixmture model introduced by Boldi & Davison (2007) """

    copula_type = CopulaTypes.DIRICHLET

    def _pickands(self, var):
        raise NotImplementedError

    def _pickandsdot(self, var, j=0):
        raise NotImplementedError

    def rextfunc(self, index):
        """ Generate from extremal Dirichlet Y follows P_x, where
        P_x is the probability of extremal functions from a Dirichlet mixture

        Input
        -----
            d     : dimension of the 1-sample.
            index : index of the location. An integer in {0, ..., d-1}.
            sigmat: a d times n dimensional vector of positive parameter
                    values for the Dirichlet vector.
            theta : a code{randinteger} vector of mixture weights, which sum to 1.

        Output
        ------
            a code{d}-vector from P_x
        """
        int_seq = np.arange(self.dim)
        # Probability weights
        weight = np.zeros(len(self.theta))
        for k in range(0, len(self.theta)):
            weight[k] = len(self.theta) * self.theta[k] * \
                self.sigmat[index, k] / sum(self.sigmat[:, k])

        randinteger = np.random.choice(int_seq, 1, False, weight)[0]

        vectnum = np.zeros(self.dim)
        vectdenum = np.random.gamma(
            self.sigmat[index, randinteger] + 1.0, 1.0, size=1)
        for j in range(0, self.dim):
            vectnum[j] = np.random.gamma(
                self.sigmat[j, randinteger], 1.0, size=1) / vectdenum
        vectnum[index] = 1.0
        return vectnum


class Bilog(Extreme):
    """ The bilogistic distribution model Smith (1990) """

    copula_type = CopulaTypes.BILOG

    def _pickands(self, var):
        raise NotImplementedError

    def _pickandsdot(self, var, j=0):
        raise NotImplementedError

    def rextfunc(self, index, normalize=True):
        """ Random variate generation for Dirichlet distribution on S_d
        A function to sample Dirichlet random variables, based on the representation
        as ratios of Gamma.

        Input
        -----
            n         : sample size
            alpha     : vector of parameter
            normalize : If code{False}, the function returns Gamma variates with parameter
                        code{alpha}.
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


def frechet(var):
    """
        Probability distribution function for Frechet's law
    """
    return np.exp(-1/var)
