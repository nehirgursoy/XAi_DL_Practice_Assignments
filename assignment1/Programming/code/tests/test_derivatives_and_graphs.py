#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
This script was written as a test module for the 'derivatives_and_graphs.ipynb'-jupyter notebook.
That notebook tests your knowledge on how to compute derivatives and gradients for Pytorch tensors.
Students may use the functions below to validate their solutions of the proposed tasks.

@author: Sebastian Doerrich
@copyright: Copyright (c) 2022, Chair of Explainable Machine Learning (xAI), Otto-Friedrich University of Bamberg
@credits: [Christian Ledig, Sebastian Doerrich]
@license: CC BY-SA
@version: 1.0
@python: Python 3
@maintainer: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
@status: Production
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
# Import packages
import numpy as np
import torch
import unittest
import argparse


# Import own files
import nbimporter  # Necessary to be able to use equations of ipynb-notebooks in python-files
import derivatives_and_graphs


class TestDerivativeSimpleFunctionAtPosition(unittest.TestCase):
    """
    The class contains all test cases for task "2.1 - Derivative of a Simple Function at a Specific Position".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the test parameters
        pass

    def test_x(self):
        """ Test for x. """

        student_version, _ = derivatives_and_graphs.derivative_of_simple_function_at_position()
        expected_version = torch.load("../data/derivatives_and_graphs/references.pt")["t1"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

    def test_gradient_fx_at_x(self):
        """ Test for f'(x) at x. """

        student_version, _ = derivatives_and_graphs.derivative_of_simple_function_at_position()
        student_version = student_version.grad.detach()
        expected_version = torch.load("../data/derivatives_and_graphs/references.pt")["t2"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

    def test_fx(self):
        """ Test for f(x). """

        _, student_version = derivatives_and_graphs.derivative_of_simple_function_at_position()
        expected_version = torch.load("../data/derivatives_and_graphs/references.pt")["t3"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")


class TestDerivativeComplicatedFunctionAtPosition(unittest.TestCase):
    """
    The class contains all test cases for task "2.2 - Derivative of a More Complicated Function at a Specific Position".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the test parameters
        pass

    def test_x(self):
        """ Test for x. """

        student_version, _ = derivatives_and_graphs.derivative_of_more_complicated_function_at_position()
        expected_version = torch.load("../data/derivatives_and_graphs/references.pt")["t4"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

    def test_gradient_gx_at_x(self):
        """ Test for g'(x) at x. """

        student_version, _ = derivatives_and_graphs.derivative_of_more_complicated_function_at_position()
        student_version = student_version.grad.detach()
        expected_version = torch.load("../data/derivatives_and_graphs/references.pt")["t5"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

    def test_gx(self):
        """ Test for g(x). """

        _, student_version = derivatives_and_graphs.derivative_of_more_complicated_function_at_position()
        expected_version = torch.load("../data/derivatives_and_graphs/references.pt")["t6"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")


class TestDerivativeEntireFunction(unittest.TestCase):
    """
    The class contains all test cases for task "2.3 - Derivative of an Entire Function".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the test parameters
        pass

    def test_x(self):
        """ Test for x. """

        student_version, _ = derivatives_and_graphs.derivative_of_relu()
        expected_version = torch.load("../data/derivatives_and_graphs/references.pt")["t7"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

    def test_gradient_fx_at_x(self):
        """ Test for f'(x) at x. """

        student_version, _ = derivatives_and_graphs.derivative_of_relu()
        student_version = student_version.grad.detach()
        expected_version = torch.load("../data/derivatives_and_graphs/references.pt")["t8"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

    def test_fx(self):
        """ Test for f(x). """

        _, student_version = derivatives_and_graphs.derivative_of_relu()
        expected_version = torch.load("../data/derivatives_and_graphs/references.pt")["t9"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")


class TestPartialDerivatives(unittest.TestCase):
    """
    The class contains all test cases for task "3 - Partial Derivatives".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the test parameters
        pass

    def test_u(self):
        """ Test for u. """

        student_version, _, _ = derivatives_and_graphs.partial_derivatives()
        expected_version = torch.load("../data/derivatives_and_graphs/references.pt")["t10"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

    def test_v(self):
        """ Test for v. """

        _, student_version, _ = derivatives_and_graphs.partial_derivatives()
        expected_version = torch.load("../data/derivatives_and_graphs/references.pt")["t11"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

    def test_gradient_fu(self):
        """ Test for f_u. """

        student_version, _, _ = derivatives_and_graphs.partial_derivatives()
        student_version = student_version.grad.detach()
        expected_version = torch.load("../data/derivatives_and_graphs/references.pt")["t12"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

    def test_gradient_fv(self):
        """ Test for f_v. """

        _, student_version, _ = derivatives_and_graphs.partial_derivatives()
        student_version = student_version.grad.detach()
        expected_version = torch.load("../data/derivatives_and_graphs/references.pt")["t13"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

    def test_fuv(self):
        """ Test for f(uv). """

        _, _, student_version = derivatives_and_graphs.partial_derivatives()
        expected_version = torch.load("../data/derivatives_and_graphs/references.pt")["t14"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")


if __name__ == '__main__':
    # Instantiate the command line parser
    parser = argparse.ArgumentParser()

    # Add the option to run only a specific test case
    parser.add_argument('--test_case', help='Name of the test case you want to run')
    parser.add_argument('--test_function', help='Name of the single test function you want to run')

    # Read the command line parameters
    args = parser.parse_args()

    # Run only a single test class
    if args.test_case:
        test_class = eval(args.test_case)

        if args.test_function:
            suite = unittest.TestSuite()
            suite.addTest(test_class(args.test_function))

        else:
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)

        unittest.TextTestRunner().run(suite)

    # Run all test classes
    else:
        unittest.main(argv=[''], verbosity=1, exit=False)
