"""Multivariate elliptical copula module contain class for sample from a multivariate
elliptical copula.
"""

import numpy as np
from scipy.stats import norm, t
from .base import Multivariate, CopulaTypes


class Gaussian(Multivariate):
    """Class for Gaussian copula
    """

    copula_type = CopulaTypes.GAUSSIAN

    def is_pos_def(self):
        """Validate the Sigma inserted.

        Raises
        ------
            ValueError : If Sigma is not positive semi definite.
        """
        if not np.all(np.linalg.eigvals(self.sigmat) > 0):
            message = "The inserted covariance matrix {} is not positive semi definite for the given {} copula."
            raise ValueError(message.format(self.sigmat, self.copula_type.name))

    def sample_unimargin(self):
        """Set margins as uniform under the segment [0,1].
        """
        self.is_pos_def()
        mean = np.zeros(self.d)
        sample = np.random.multivariate_normal(
            mean=mean, cov=self.sigmat, size=self.n_sample)
        return norm.cdf(sample)


class Student(Multivariate):
    """Class for Student copula
    """

    copula_type = CopulaTypes.STUDENT
    theta_interval = [0, float('inf')]
    invalid_thetas = [0]

    def is_pos_def(self):
        """Validate the Sigma inserted.

        Raises
        ------
            ValueError : If Sigma is not positive semi definite.
        """
        if not np.all(np.linalg.eigvals(self.sigmat) > 0):
            message = "The inserted covariance matrix {} is not positive semi definite for the given {} copula."
            raise ValueError(message.format(
                self.sigmat, self.copula_type.name))

    def multivariatet(self):
        '''
        Output:
        Produce n_sample samples of d-dimensional multivariate t distribution
        '''
        gamma = np.tile(np.random.gamma(self.theta/2., 2. /
                        self.theta, self.n_sample), (self.d, 1)).T
        normobs = np.random.multivariate_normal(
            np.zeros(self.d), self.Sigma, self.n_sample)
        return normobs/np.sqrt(gamma)

    def sample_unimargin(self):
        """Set margins as uniform under the segment [0,1].
        """
        sample = self.multivariatet()
        return t.cdf(sample, self.theta)
