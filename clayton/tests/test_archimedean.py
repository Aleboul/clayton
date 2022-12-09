"""Test file for base
"""
import unittest

import clayton
from clayton.rng.archimedean import Clayton
from clayton.rng.archimedean import Frank
from clayton.rng.archimedean import Amh
from clayton.rng.archimedean import Joe
from clayton.rng.archimedean import Nelsen9
from clayton.rng.archimedean import Nelsen10
from clayton.rng.archimedean import Nelsen11
from clayton.rng.archimedean import Nelsen12
from clayton.rng.archimedean import Nelsen13
from clayton.rng.archimedean import Nelsen14
from clayton.rng.archimedean import Nelsen15
from clayton.rng.archimedean import Nelsen22


class TestUser(unittest.TestCase):
    """A simple test class for evd copula
    """

    def test_multivariate_instantiation_clayton(self):
        """Check if initiated object does verify basic contraints
        of a Frank copula
        """
        cla1 = Clayton(
            theta=-0.5, dim=2)
        self.assertTrue(isinstance(
            cla1, clayton.rng.base.Archimedean))
        with self.assertRaises(Exception):
            Clayton(theta=-5)
        with self.assertRaises(Exception):
            # can't sample from negative theta where dim > 2
            cla1 = Clayton(theta=-2, dim=3, n_sample=1000)
            cla1.sample_unimargin()

        # sampling test

        n_sample, dim, theta = 1000, 100, 0.5
        cla1 = Clayton(theta=theta, dim=dim, n_sample=n_sample)
        cla1.sample_unimargin()

    def test_multivariate_instantiation_frank(self):
        """Check if initiated object does verify basic contraints
        of a Frank copula
        """
        fra1 = Frank(
            theta=-0.5, dim=2)
        self.assertTrue(isinstance(
            fra1, clayton.rng.base.Archimedean))
        with self.assertRaises(Exception):
            Frank(theta=0.0)
        with self.assertRaises(Exception):
            # can't sample from negative theta where dim > 2
            fra1 = Frank(theta=-2, dim=3, n_sample=1000)
            fra1.sample_unimargin()

        # sampling test

        n_sample, dim, theta = 1000, 100, 0.5
        fra1 = Frank(theta=theta, dim=dim, n_sample=n_sample)
        fra1.sample_unimargin()

    def test_multivariate_instantiation_amh(self):
        """Check if initiated object does verify basic contraints
        of a AMH copula
        """
        amh1 = Amh(
            theta=-0.5, dim=2)
        self.assertTrue(isinstance(
            amh1, clayton.rng.base.Archimedean))
        with self.assertRaises(Exception):
            Amh(theta=-5.0)
        with self.assertRaises(Exception):
            # can't sample from negative theta where dim > 2
            amh1 = Amh(theta=-0.5, dim=3, n_sample=1000)
            amh1.sample_unimargin()

        # sampling test

        n_sample, dim, theta = 1000, 100, 0.5
        amh1 = Amh(theta=theta, dim=dim, n_sample=n_sample)
        amh1.sample_unimargin()

    def test_multivariate_instantiation_joe(self):
        """Check if initiated object does verify basic contraints
        of a Joe copula
        """
        joe1 = Joe(
            theta=1.5, dim=2)
        self.assertTrue(isinstance(
            joe1, clayton.rng.base.Archimedean))
        with self.assertRaises(Exception):
            Joe(theta=-5.0)

        # sampling test

        n_sample, dim, theta = 1000, 100, 2
        joe1 = Joe(theta=theta, dim=dim, n_sample=n_sample)
        joe1.sample_unimargin()

    def test_multivariate_instantiation_nelsen9(self):
        """Check if initiated object does verify basic contraints
        of a Nelsen9 copula
        """
        nel9 = Nelsen9(
            theta=0.5, dim=2)
        self.assertTrue(isinstance(
            nel9, clayton.rng.base.Archimedean))
        with self.assertRaises(Exception):
            Nelsen9(theta=-5.0)

        # sampling test

        n_sample, dim, theta = 1000, 2, 0.5
        nel9 = Nelsen9(theta=theta, dim=dim, n_sample=n_sample)
        nel9.sample_unimargin()

    def test_multivariate_instantiation_nelsen10(self):
        """Check if initiated object does verify basic contraints
        of a Nelsen10 copula
        """
        nel10 = Nelsen10(
            theta=0.5, dim=2)
        self.assertTrue(isinstance(
            nel10, clayton.rng.base.Archimedean))
        with self.assertRaises(Exception):
            Nelsen10(theta=-5.0)

        # sampling test

        n_sample, dim, theta = 1000, 2, 0.5
        nel10 = Nelsen10(theta=theta, dim=dim, n_sample=n_sample)
        nel10.sample_unimargin()

    def test_multivariate_instantiation_nelsen11(self):
        """Check if initiated object does verify basic contraints
        of a Nelsen11 copula
        """
        nel11 = Nelsen11(
            theta=0.4, dim=2)
        self.assertTrue(isinstance(
            nel11, clayton.rng.base.Archimedean))
        with self.assertRaises(Exception):
            Nelsen11(theta=-5.0)

        # sampling test

        n_sample, dim, theta = 1000, 2, 0.4
        nel11 = Nelsen11(theta=theta, dim=dim, n_sample=n_sample)
        nel11.sample_unimargin()

    def test_multivariate_instantiation_nelsen12(self):
        """Check if initiated object does verify basic contraints
        of a Nelsen12 copula
        """
        nel12 = Nelsen12(
            theta=0.5, dim=2)
        self.assertTrue(isinstance(
            nel12, clayton.rng.base.Archimedean))
        with self.assertRaises(Exception):
            Nelsen12(theta=-5.0)
        with self.assertRaises(Exception):
            # can't sample from where dim > 2
            nel12 = Nelsen12(theta=0.5, dim=3, n_sample=1000)
            nel12.sample_unimargin()

        # sampling test

        n_sample, dim, theta = 1000, 2, 0.5
        nel12 = Nelsen12(theta=theta, dim=dim, n_sample=n_sample)
        nel12.sample_unimargin()

    def test_multivariate_instantiation_nelsen13(self):
        """Check if initiated object does verify basic contraints
        of a Nelsen13 copula
        """
        nel13 = Nelsen13(
            theta=0.5, dim=2)
        self.assertTrue(isinstance(
            nel13, clayton.rng.base.Archimedean))
        with self.assertRaises(Exception):
            Nelsen13(theta=-5.0)
        with self.assertRaises(Exception):
            # can't sample from where dim > 2
            nel13 = Nelsen13(theta=0.5, dim=3, n_sample=1000)
            nel13.sample_unimargin()

        # sampling test

        n_sample, dim, theta = 1000, 2, 0.5
        nel13 = Nelsen13(theta=theta, dim=dim, n_sample=n_sample)
        nel13.sample_unimargin()

    def test_multivariate_instantiation_nelsen14(self):
        """Check if initiated object does verify basic contraints
        of a Nelsen14 copula
        """
        nel14 = Nelsen14(
            theta=1.5, dim=2)
        self.assertTrue(isinstance(
            nel14, clayton.rng.base.Archimedean))
        with self.assertRaises(Exception):
            Nelsen14(theta=1.0)
        with self.assertRaises(Exception):
            # can't sample from where dim > 2
            nel14 = Nelsen14(theta=2.5, dim=3, n_sample=1000)
            nel14.sample_unimargin()

        # sampling test

        n_sample, dim, theta = 1000, 2, 1.5
        nel14 = Nelsen14(theta=theta, dim=dim, n_sample=n_sample)
        nel14.sample_unimargin()

    def test_multivariate_instantiation_nelsen15(self):
        """Check if initiated object does verify basic contraints
        of a Nelsen15 copula
        """
        nel15 = Nelsen15(
            theta=1.5, dim=2)
        self.assertTrue(isinstance(
            nel15, clayton.rng.base.Archimedean))
        with self.assertRaises(Exception):
            Nelsen15(theta=-0.5)
        with self.assertRaises(Exception):
            # can't sample from where dim > 2
            nel15 = Nelsen15(theta=2.5, dim=3, n_sample=1000)
            nel15.sample_unimargin()

        # sampling test

        n_sample, dim, theta = 1000, 2, 1.5
        nel15 = Nelsen15(theta=theta, dim=dim, n_sample=n_sample)
        nel15.sample_unimargin()

    def test_multivariate_instantiation_nelsen22(self):
        """Check if initiated object does verify basic contraints
        of a Nelsen22 copula
        """
        nel22 = Nelsen22(
            theta=0.5, dim=2)
        self.assertTrue(isinstance(
            nel22, clayton.rng.base.Archimedean))
        with self.assertRaises(Exception):
            Nelsen22(theta=1.5)
        with self.assertRaises(Exception):
            # can't sample from where dim > 2
            nel22 = Nelsen22(theta=0.75, dim=3, n_sample=1000)
            nel22.sample_unimargin()

        # sampling test

        n_sample, dim, theta = 1000, 2, 0.5
        nel22 = Nelsen22(theta=theta, dim=dim, n_sample=n_sample)
        nel22.sample_unimargin()


if __name__ == '__main__':
    unittest.main()
