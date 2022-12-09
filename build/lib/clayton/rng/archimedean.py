"""Let :math:`\phi` be a generator which is a strictly decreasing, convex function
from :math:`[0,1]` to :math:`[0,\infty]` such that :math:`\phi(1)=0` and :math:`\phi(1)=\infty`.
We denote by :math:`\phi^{-1}` the generalized inverse of :math:`\phi`. Let denote by

.. math:: C(u_1,\dots,u_d) = \phi^{-1} ( \phi(u_1),\dots, \phi(u_d)).

If this relation holds and :math:`C` is a copula function, then :math:`C` is called
and Archimedean copula.

- Archimedean copula (:py:class:`Archimedean`) from :py:mod:`clayton.rng.base`
    - Ali Mikhail Haq copula (:py:class:`Amh`)
    - Clayton model (:py:class:`Clayton`)
    - Frank copula(:py:class:`Frank`)
    - Joe (:py:class:`Joe`)
    - Nelsen10 (:py:class:`Nelsen10`)
    - Nelsen11 (:py:class:`Nelsen11`)
    - Nelsen12 (:py:class:`Nelsen12`)
    - Nelsen13 (:py:class:`Nelsen13`)
    - Nelsen14 (:py:class:`Nelsen14`)
    - Nelsen15 (:py:class:`Nelsen15`)
    - Nelsen22 (:py:class:`Nelsen22`)
    - Nelsen9 (:py:class:`Nelsen9`)
"""
# pylint: disable=too-few-public-methods

import math
import numpy as np
from .utils import rSibuya_vec_c, rLogarithmic
from .base import CopulaTypes, Archimedean


class Clayton(Archimedean):
    """Class for Clayton copula.

    Args:
        Archimedean (object):
            Archimedean object.

    Returns:
        clayton.rng.archimedean.logistic
    """

    copula_type = CopulaTypes.CLAYTON
    theta_interval = [-1.0, float('inf')]
    invalid_thetas = [0]

    def _generator(self, var):
        """Return the generator function.
        .. math:: phi(t) = frac{1}{theta} ( t^{-theta}-1), 0 < t < 1.

        Args:
            var (float):
                value to evaluate the generator function.

        Returns:
            float.
        """
        return (1.0 / self.theta) * (np.power(var, -self.theta) - 1)

    def _generator_inv(self, var):
        """Return the generator inverse.
        .. math:: phi^leftarrow(u) = (1+theta*t)^{-1/theta},  t >= 0.

        Args:
            var (float):
                value to evaluate the generator inverse.

        Returns:
            float
        """
        return np.power((1.0 + self.theta*var), -1/self.theta)

    def _generator_dot(self, var):
        """Return the derivative of the generator function
        .. math:: phi'(t) = -(t)^{-theta-1},  0 < t < 1.

        Args:
            var (float):
                value to evaluate the derivative of the generator

        Returns:
            float
        """
        return -np.power(var, -self.theta-1)

    def _rfrailty(self):
        """Sample from frailty distribution, a Gamma
        with parameters (1/theta,1).

        Returns:
            float
        """
        return np.random.gamma(1/self.theta, 1, 1)


class Frank(Archimedean):
    """Class for Frank copula model.

    Args:
        Archimedean (object):
            Archimedean object.

    Returns:
        clayton.rng.archimidean.Frank
    """

    copula_type = CopulaTypes.FRANK
    theta_interval = [-float('inf'), float('inf')]
    invalid_thetas = [0.0]

    def _generator(self, var):
        """Return the generator function.
        .. math:: phi(t) = 1/theta(t^{-theta}-1),  0 < t < 1.

        Args:
            var (float):
                value to evaluate the generator function.

        Returns:
            float.
        """
        return -np.log((np.exp(-self.theta*var)-1) / (np.exp(-self.theta)-1))

    def _generator_inv(self, var):
        """Return the generator inverse.
        .. math:: phi^leftarrow(t) = (1-theta) / (exp(t) - theta),  t >= 0.

        Args:
            var (float):
                value to evaluate the generator inverse.

        Returns:
            float
        """
        return -(1 / self.theta)*np.log(1+np.exp(-var)*(np.exp(-self.theta)-1))

    def _generator_dot(self, var):
        """Return the derivative of the generator function
        .. math:: phi'(t) = theta(-ln(t))^theta / (t*ln(t)).

        Args:
            var (float):
                value to evaluate the derivative of the generator

        Returns:
            float
        """
        value_1 = self.theta * np.exp(-self.theta * var)
        value_2 = np.exp(-self.theta*var) - 1
        return value_1 / value_2

    def _rfrailty(self):
        """Sample from frailty distribution, a Logarithmic with
        parameter 1-exp(-theta).

        Returns:
            float
        """

        output = rLogarithmic(1-math.exp(-self.theta))

        return output


class Amh(Archimedean):
    """Class for Amh copula model.

    Args:
        Archimedean (object):
            Archimedean object.

    Returns:
        clayton.rng.archimedean.Amh
    """

    copula_type = CopulaTypes.AMH
    theta_interval = [-1.0, 1.0]
    invalid_thetas = []

    def _generator(self, var):
        """Return the generator function.
        .. math:: phi(t) = log(frac{1-theta*(1-t)}{t}),  0 < t < 1.

        Args:
            var (float):
                value to evaluate the generator function.

        Returns:
            float.
        """
        return np.log((1-self.theta*(1-var)) / var)

    def _generator_inv(self, var):
        """Return the generator inverse.
        .. math:: (1-theta) / (exp(t) - theta),  t >= 0.

        Args:
            var (float):
                value to evaluate the generator inverse.

        Returns:
            float
        """
        value_1 = 1-self.theta
        value_2 = np.exp(var) - self.theta
        value_ = value_1 / value_2
        return value_

    def _generator_dot(self, var):
        """Return the derivative of the generator function
        .. math:: phi'(t) = frac{theta-1}{t(1-theta*(1-t))},  0 < t < 1.

        Args:
            var (float):
                value to evaluate the derivative of the generator

        Returns:
            float
        """
        value_1 = self.theta-1
        value_2 = var*(1-self.theta*(1-var))
        return value_1 / value_2

    def _rfrailty(self):
        """Sample from the frailty distribution, a geometric law
        with parameter 1-theta.

        Returns:
            float
        """
        output = np.random.geometric(1-self.theta, 1)

        return output


class Joe(Archimedean):
    """Class for Joe copula model.

    Args:
        Archimedean (object):
            Archimedean object.

    Returns:
        clayton.rng.archimedean.Joe
    """

    copula_type = CopulaTypes.JOE
    theta_interval = [1, float('inf')]
    invalid_thetas = []

    def _generator(self, var):
        """Return the generator function.
        .. math:: phi(t) = -log(1-(1-t)^theta),  0 < t < 1.

        Args:
            var (float):
                value to evaluate the generator function.

        Returns:
            float.
        """
        return -np.log(1-np.power(1-var, self.theta))

    def _generator_inv(self, var):
        """Return the generator inverse.
        .. math:: phi^leftarrow(t) = 1-(1-exp(-t))^{1/theta},  t >= 0.

        Args:
            var (float):
                value to evaluate the generator inverse.

        Returns:
            float
        """
        return 1 - np.power(1-np.exp(-var), 1/self.theta)

    def _generator_dot(self, var):
        """Return the derivative of the generator function
        .. math:: phi'(t) = -(t)^{-theta-1},  0 < t < 1.

        Args:
            var (float):
                value to evaluate the derivative of the generator

        Returns:
            float
        """
        value_1 = -self.theta * np.power(1-var, self.theta-1)
        value_2 = 1 - np.power(1-var, self.theta)
        return value_1 / value_2

    def _rfrailty(self):
        """Sample from the frailty distribution, a Sibuya law
        with parameters (1,1/theta).

        Returns:
            float
        """
        output = rSibuya_vec_c(1, 1/self.theta)

        return output


class Nelsen9(Archimedean):
    """Class for Nelsen9 copula model.

    Args:
        Archimedean (object):
            Archimedean object.

    Returns:
        copula.rng.archimedean.Nelsen9
    """

    copula_type = CopulaTypes.NELSEN9
    theta_interval = [0, 1]
    invalid_thetas = [0]

    def _generator(self, var):
        """Return the generator function.
        .. math:: phi(t) = log(1-theta*log(t)),  0 < t < 1.

        Args:
            var (float):
                value to evaluate the generator function.

        Returns:
            float.
        """
        return np.log(1-self.theta*np.log(var))

    def _generator_inv(self, var):
        """Return the generator inverse.
        .. math:: phi^leftarrow(t) = exp(frac{1-exp(t)}{theta}),  t >= 0.

        Args:
            var (float):
                value to evaluate the generator inverse.

        Returns:
            float
        """
        return np.exp((1-np.exp(var))/self.theta)

    def _generator_dot(self, var):
        """Return the derivative of the generator function
        .. math:: phi'(t) = frac{theta}{t * (theta*log(t) - 1)},  0 < t < 1.

        Args:
            var (float):
                value to evaluate the derivative of the generator

        Returns:
            float
        """
        value_1 = self.theta
        value_2 = var * (self.theta * np.log(var) - 1)
        return value_1 / value_2


class Nelsen10(Archimedean):
    """Class for Nelsen10 copula model.

    Args:
        Archimedean (object):
            Archimedean object.

    Returns:
        clayton.rng.archimedean.nelsen10
    """

    copula_type = CopulaTypes.NELSEN10
    theta_interval = [0, 1]
    invalid_thetas = [0]

    def _generator(self, var):
        """Return the generator function.
        .. math:: phi(t) = log(2*(t)^{-theta}-1),  0 < t < 1.

        Args:
            var (float):
                value to evaluate the generator function.

        Returns:
            float.
        """
        return np.log(2*np.power(var, -self.theta)-1)

    def _generator_inv(self, var):
        """Return the generator inverse.
        .. math:: phi^leftarrow(t) = (frac{exp(t)+1}{2})^{-1/theta},  t >= 0.

        Args:
            var (float):
                value to evaluate the generator inverse.

        Returns:
            float
        """
        value_1 = np.exp(var) + 1
        value_2 = 2
        return np.power(value_1 / value_2, -1/self.theta)

    def _generator_dot(self, var):
        """Return the derivative of the generator function
        .. math:: phi'(t) = frac{-2*theta*(t)^{-1-theta}}{2t^{-theta}-1},
                            0 < t < 1.
        Args:
            var (float):
                value to evaluate the derivative of the generator

        Returns:
            float
        """
        value_1 = -self.theta*2*np.power(var, -1-self.theta)
        value_2 = 2*np.power(var, -self.theta)-1
        return value_1 / value_2


class Nelsen11(Archimedean):
    """Class for Nelsen11 copula model.

    Args:
        Archimedean (object):
            Archimedean object.

    Returns:
        clayton.rng.archimedean.Nelsen11
    """

    copula_type = CopulaTypes.NELSEN11
    theta_interval = [0, 0.5]
    invalid_thetas = [0.0]

    def _generator(self, var):
        """Return the generator function.
        .. math:: phi(t) = log(2-t^{theta}),  0 < t < 1.

        Args:
            var (float):
                value to evaluate the generator function.

        Returns:
            float.
        """
        return np.log(2-np.power(var, self.theta))

    def _generator_inv(self, var):
        """Return the generator inverse.
        .. math::  phi^leftarrow(t) = (2 - exp(t))^{1/theta},  t >= 0.

        Args:
            var (float):
                value to evaluate the generator inverse.

        Returns:
            float
        """
        return np.power(2-np.exp(var), 1/self.theta)

    def _generator_dot(self, var):
        """Return the derivative of the generator function
        .. math:: phi'(t) = frac{-theta * t^{theta - 1}}{2 - t^theta},  0 < t < 1.

        Args:
            var (float):
                value to evaluate the derivative of the generator

        Returns:
            float
        """
        value_1 = -self.theta*np.power(var, self.theta-1)
        value_2 = 2 - np.power(var, self.theta)
        return value_1 / value_2


class Nelsen12(Archimedean):
    """Class for Nelsen12 copula model.

    Args:
        Archimedean (object):
            Archimedean object.

    Returns:
        clayton.rng.archimedean.Nelsen12
    """

    copula_type = CopulaTypes.NELSEN12
    theta_interval = [0, float('inf')]
    invalid_thetas = [0]

    def _generator(self, var):
        """Return the generator function.
        .. math:: phi(t) = (frac{1}{t} - 1)^theta,  0 < t < 1.

        Args:
            var (float):
                value to evaluate the generator function.

        Returns:
            float.
        """
        return np.power(1/var - 1, self.theta)

    def _generator_inv(self, var):
        """Return the generator inverse.
        .. math::  phi^leftarrow(t) = (1+t^{1/theta})^{-1},  t >= 0.

        Args:
            var (float):
                value to evaluate the generator inverse.

        Returns:
            float
        """
        value_1 = 1 + np.power(var, 1/self.theta)
        return np.power(value_1, -1)

    def _generator_dot(self, var):
        """Return the derivative of the generator function
        .. math:: phi'(t) = theta * (-1/t^2) * (1/t - 1)^{theta - 1},  0 < t < 1.

        Args:
            var (float):
                value to evaluate the derivative of the generator

        Returns:
            float
        """
        value_ = self.theta*(-1/np.power(var, 2)) * \
            np.power(1/var - 1, self.theta - 1)
        return value_


class Nelsen13(Archimedean):
    """Class for Nelsen13 copula model.

    Args:
        Archimedean (object):
            Archimedean object.

    Returns:
        clayton.rng.archimedean.Nelsen13
    """

    copula_type = CopulaTypes.NELSEN13
    theta_interval = [0, float('inf')]
    invalid_thetas = [0]

    def _generator(self, var):
        """Return the generator function.
        .. math:: phi(t) = (1-log(t))^{theta}-1,  0 < t < 1.

        Args:
            var (float):
                value to evaluate the generator function.

        Returns:
            float.
        """
        return np.power(1-np.log(var), self.theta)-1

    def _generator_inv(self, var):
        """Return the generator inverse.
        .. math:: 1 - (t+1)^{1/theta},  t >= 0.

        Args:
            var (float):
                value to evaluate the generator inverse.

        Returns:
            float
        """
        value_1 = 1-np.power(var+1, 1/self.theta)
        return np.exp(value_1)

    def _generator_dot(self, var):
        """Return the derivative of the generator function
        .. math:: phi'(t) = -theta * (1-log(t))^{theta-1}/t,  0 < t < 1.
        Args:
            var (float):
                value to evaluate the derivative of the generator

        Returns:
            float
        """
        value_ = -self.theta*np.power(1-np.log(var), self.theta-1) / var
        return value_


class Nelsen14(Archimedean):
    """Class for Nelsen14 copula model

    Args:
        Archimedean (object):
            Archimedean object.

    Returns:
        clayton.rng.archimedean.Nelsen14
    """

    copula_type = CopulaTypes.NELSEN14
    theta_interval = [1.0, float('inf')]
    invalid_thetas = [1.0]

    def _generator(self, var):
        """Return the generator function.
        .. math:: phi(t) = (t^{-1/theta}-1)^theta,  0 < t < 1.

        Args:
            var (float):
                value to evaluate the generator function.

        Returns:
            float.
        """
        return np.power(np.power(var, -1/self.theta)-1, self.theta)

    def _generator_inv(self, var):
        """Return the generator inverse.
        .. math:: (t^{1/theta}+1)^{-theta},  t >= 0.

        Args:
            var (float):
                value to evaluate the generator inverse.

        Returns:
            float
        """
        return np.power(np.power(var, 1/self.theta)+1, -self.theta)

    def _generator_dot(self, var):
        """Return the derivative of the generator function
        .. math:: phi'(t) = frac{-(t)^{-theta-1}}{t^{-1/theta}-1},  0 < t < 1.
        Args:
            var (float):
                value to evaluate the derivative of the generator

        Returns:
            float
        """
        value_1 = -np.power(var, -self.theta - 1)
        value_2 = np.power(var, -1/self.theta)-1
        return value_1 * np.power(value_2, self.theta-1)


class Nelsen15(Archimedean):
    """Class for Nelsen15 copula model.

    Args:
        Archimedean (object):
            Archimedean object

    Returns:
        clayton.rng.archimedean.Nelsen15
    """

    copula_type = CopulaTypes.NELSEN15
    theta_interval = [1, float('inf')]
    invalid_thetas = []

    def _generator(self, var):
        """Return the generator function.
        .. math:: phi(t) = (1-t^{1/theta})^theta,  0 < t < 1.

        Args:
            var (float):
                value to evaluate the generator function.

        Returns:
            float.
        """
        return np.power(1-np.power(var, 1/self.theta), self.theta)

    def _generator_inv(self, var):
        """Return the generator inverse.
        .. math::  phi^leftarrow(t) = (1-t^{1/theta})^theta,  t >= 0.

        Args:
            var (float):
                value to evaluate the generator inverse.

        Returns:
            float
        """
        return np.power(1-np.power(var, 1/self.theta), self.theta)

    def _generator_dot(self, var):
        """Return the derivative of the generator function
        .. math:: phi'(t) = -(t)^{1/theta-1}(1-t^theta)^{theta-1},  0 < t < 1.
        Args:
            var (float):
                value to evaluate the derivative of the generator

        Returns:
            float
        """
        value_ = - np.power(var, 1/self.theta - 1) * \
            np.power(1-np.power(var, self.theta), self.theta - 1)
        return value_


class Nelsen22(Archimedean):
    """Class for Nelsen22 copula model.

    Args:
        Archimedean (object):
            Archimedean object.

    Returns:
        clayton.rng.archimedean.Nelsen22
    """

    copula_type = CopulaTypes.NELSEN22
    theta_interval = [0, 1]
    invalid_thetas = []

    def _generator(self, var):
        """Return the generator function.
        .. math:: phi(t) = arcsin(1-t^theta),  0 < t < 1.

        Args:
            var (float):
                value to evaluate the generator function.

        Returns:
            float.
        """
        return np.arcsin(1-np.power(var, self.theta))

    def _generator_inv(self, var):
        """Return the generator inverse.
        .. math:: (1-sin(t))^{1/theta},  t >= 0.

        Args:
            var (float):
                value to evaluate the generator inverse.

        Returns:
            float
        """
        return np.power(1-np.sin(var), 1/self.theta)

    def _generator_dot(self, var):
        """Return the derivative of the generator function
        .. math:: phi'(t) = frac{-theta*(t)^{theta-1}}{(1-(t^theta-1)^2)^0.5}

        Args:
            var (float):
                value to evaluate the derivative of the generator

        Returns:
            float
        """
        value_1 = - self.theta * np.power(var, self.theta-1)
        value_2 = np.power(1-np.power(np.power(var, self.theta)-1, 2), 1/2)
        return value_1 / value_2
