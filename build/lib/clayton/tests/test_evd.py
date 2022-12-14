"""Test file for base
"""
import unittest
import numpy as np

import clayton
from clayton.rng.evd import Logistic
from clayton.rng.evd import AsymmetricLogistic
from clayton.rng.evd import HuslerReiss
from clayton.rng.evd import AsyNegLog
from clayton.rng.evd import AsyMix
from clayton.rng.evd import TEV
from clayton.rng.evd import Dirichlet
from clayton.rng.evd import Bilog


class TestUser(unittest.TestCase):
    """A simple test class for evd copula
    """

    def test_multivariate_instantiation_logistic(self):
        """Check if initiated object does verify basic contraints
        of a logistic copula
        """
        log1 = Logistic(
            theta=1, dim=5)
        self.assertTrue(isinstance(
            log1, clayton.rng.base.Extreme))
        with self.assertRaises(Exception):
            Logistic(theta=-1)
        with self.assertRaises(Exception):
            Logistic(theta=complex(1, 2))
        with self.assertRaises(Exception):
            Logistic(theta=50)

        # sampling test

        n_sample, dim, theta = 1000, 100, 0.5
        log1 = Logistic(theta=theta, dim=dim, n_sample=n_sample)
        log1.sample_unimargin()

    def test_multivariate_instantiation_asylog(self):
        """Check if initiated object basic contraints
        of a asymmetric logistic copula
        """
        al1 = AsymmetricLogistic(
            theta=1/4, asy=[0.05, 0.3, [0.95, 0.7]], dim=2)
        self.assertTrue(isinstance(
            al1, clayton.rng.base.Extreme))
        with self.assertRaises(Exception):
            # negative theta
            AsymmetricLogistic(theta=-1, asy=[0.05, 0.3, [0.95, 0.7]])
        with self.assertRaises(Exception):
            AsymmetricLogistic(
                theta=1/4, asy=[0.5, 0.3, [0.95, 0.7]], dim=2)  # does not sum to 1

        n_sample, dim, theta, asy = 1000, 2, 1/4, [0.05, 0.3, [0.95, 0.7]]
        al1 = AsymmetricLogistic(
            theta=theta, asy=asy, dim=dim, n_sample=n_sample)
        al1.sample_unimargin()

    def test_multivariate_instantiation_huslerreiss(self):
        """Check if initiated object basic contraints
        of an Husler Reiss copula
        """
        sigmat = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        hr1 = HuslerReiss(sigmat=sigmat, dim=3)
        self.assertTrue(isinstance(
            hr1, clayton.rng.base.Extreme))
        with self.assertRaises(Exception):
            HuslerReiss(sigmat=np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), dim=3)  # not a cnsd

        n_sample, dim, sigmat = 1000, 3, np.array(
            [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        hr1 = HuslerReiss(sigmat=sigmat, dim=dim, n_sample=n_sample)
        hr1.sample_unimargin()

    def test_multivariate_instantiation_asyneglog(self):
        """Check if initiated object basic contraints
        of an asymmetric negative logistic copula
        """
        theta, psi1, psi2 = 10, 0.5, 1.0
        anl1 = AsyNegLog(theta=theta, psi1=psi1, psi2=psi2)
        self.assertTrue(isinstance(
            anl1, clayton.rng.base.Extreme))
        with self.assertRaises(Exception):
            theta = -2.0  # wrong theta
            AsyNegLog(theta=theta, psi1=psi1, psi2=psi2)
        with self.assertRaises(Exception):
            psi1 = 1.3  # wrong psi1
            AsyNegLog(theta=theta, psi1=psi1, psi2=psi2)
        with self.assertRaises(Exception):
            theta = 1.5  # wrong psi2
            AsyNegLog(theta=theta, psi1=psi1, psi2=psi2)
        n_sample, dim, theta, psi1, psi2 = 1000, 2, 10, 0.5, 1.0
        anl1 = AsyNegLog(theta=theta, psi1=psi1, psi2=psi2,
                         n_sample=n_sample, dim=dim)
        anl1.sample_unimargin()

    def test_multivariate_instantiation_asymix(self):
        """Check if initiated object basic contraints
        of an asymmetric mixed copula
        """
        theta, psi1 = 4/3, -1/3
        am1 = AsyMix(theta=theta, psi1=psi1)
        self.assertTrue(isinstance(
            am1, clayton.rng.base.Extreme))
        with self.assertRaises(Exception):
            theta = -2.0  # wrong theta
            AsyMix(theta=theta, psi1=psi1)
        with self.assertRaises(Exception):
            theta, psi1 = 5/3, -1/3  # first inequality is wrong
            AsyMix(theta=theta, psi1=psi1)
        with self.assertRaises(Exception):
            theta, psi1 = 0.2, 0.5  # third inequality is wrong
            AsyMix(theta=theta, psi1=psi1)
        n_sample, dim, theta, psi1 = 1000, 2, 4/3, -1/3
        am1 = AsyMix(theta=theta, psi1=psi1, n_sample=n_sample, dim=dim)
        am1.sample_unimargin()

    def test_multivariate_instantiation_tev(self):
        """Check if initiated object basic contraints
        of t-copula
        """
        sigmat, psi1 = np.array([[1, 0.8], [0.8, 1]]), 0.2
        tv1 = TEV(sigmat=sigmat, psi1=psi1)
        self.assertTrue(isinstance(
            tv1, clayton.rng.base.Extreme))
        with self.assertRaises(Exception):
            psi1 = -2  # wrong psi1
            TEV(sigmat=sigmat, psi1=psi1)
        with self.assertRaises(Exception):
            # not a symmetric matrix
            sigmat = np.array([[1, 0.7], [0.8, 1.0]])
            TEV(sigmat=sigmat, psi1=psi1)
        n_sample, dim, sigmat, psi1 = 1000, 3, np.array(
            [[1, 0.8, 0.8], [0.8, 1.0, 0.8], [0.8, 0.8, 1.0]]), 0.2
        tv1 = TEV(sigmat=sigmat, psi1=psi1, n_sample=n_sample, dim=dim)
        tv1.sample_unimargin()

    def test_multivariate_instantiation_dir(self):
        """Check if initiated object basic contraints
        of dirichlet mixture models
        """
        sigmat = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        theta = np.array([1/3, 1/3, 1/3])
        dr1 = Dirichlet(sigmat=sigmat, theta=theta, dim=3)
        self.assertTrue(isinstance(
            dr1, clayton.rng.base.Extreme))
        with self.assertRaises(Exception):
            theta = np.array([1/3, 2/3, 1/3])  # wrong theta
            Dirichlet(sigmat=sigmat, theta=theta, dim=3)
        with self.assertRaises(Exception):
            # wrong matrix
            sigmat = np.array([[1, 0.7], [0.8, 1.0]])
            Dirichlet(sigmat=sigmat, theta=theta, dim=3)
        n_sample, dim, sigmat, theta = 1000, 3, np.array(
            [[2, 1, 1], [1, 2, 1], [1, 1, 2]]), np.array([1/3, 1/3, 1/3])
        dr1 = Dirichlet(sigmat=sigmat, theta=theta, n_sample=n_sample, dim=dim)
        dr1.sample_unimargin()

    def test_multivariate_instantiation_bilog(self):
        """Check if initiated object basic contraints
        of t-copula
        """
        theta = np.array([0.5, 0.3, 0.8, 0.9])
        bl1 = Bilog(theta=theta, dim=4)
        self.assertTrue(isinstance(
            bl1, clayton.rng.base.Extreme))
        with self.assertRaises(Exception):
            theta = np.array([1.2, 0.3, 0.8, 0.9])  # wrong psi1
            Bilog(theta=theta, dim=4)
        n_sample, dim, theta = 1000, 4, np.array([0.5, 0.3, 0.8, 0.9])
        bl1 = Bilog(theta=theta, dim=dim, n_sample=n_sample)
        bl1.sample_unimargin()


if __name__ == '__main__':
    unittest.main()
