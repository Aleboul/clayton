"""Test file for base
"""

import unittest

import clayton
from clayton.rng.base import Multivariate
from clayton.rng.base import Archimedean
from clayton.rng.base import Extreme


class TestUser(unittest.TestCase):
    """A simple test class
    """

    def test_multivariate_instantiation(self):
        """Check if initiated object is indeed a multivariate object
        """
        multivariate1 = Multivariate()
        self.assertTrue(isinstance(
            multivariate1, clayton.rng.base.Multivariate))

    def test_archimedean_instantiation(self):
        """Check if initiated object is indeed an archimedean object
        """
        archimedean1 = Archimedean()
        self.assertTrue(isinstance(
            archimedean1, clayton.rng.base.Multivariate))
        self.assertTrue(isinstance(archimedean1, clayton.rng.base.Archimedean))

    def test_extreme_instantiation(self):
        """Check if initiated object is indeed an extreme object
        """
        extreme1 = Extreme()
        self.assertTrue(isinstance(
            extreme1, clayton.rng.base.Multivariate))
        self.assertTrue(isinstance(extreme1, clayton.rng.base.Extreme))

    def test_multivariate_misspec_dim(self):
        """Check if the object does not instantiate with an incorrect
        dimension
        """
        with self.assertRaises(Exception):
            Multivariate(dim=complex(1, 2))
        with self.assertRaises(Exception):
            Extreme(dim=0.5)
        with self.assertRaises(Exception):
            Archimedean(dim=-1)

    def test_multivariate_misspec_nsample(self):
        """Check if the object does not instantiate with an incorrect
        sample size
        """
        with self.assertRaises(Exception):
            Multivariate(n_sample=complex(1, 2))
        with self.assertRaises(Exception):
            Extreme(n_sample=0.5)
        with self.assertRaises(Exception):
            Archimedean(n_sample=-1)


if __name__ == '__main__':
    unittest.main()
