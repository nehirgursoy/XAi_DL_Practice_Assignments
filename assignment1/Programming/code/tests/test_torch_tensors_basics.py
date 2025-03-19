#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
This script was written as a test module for the 'torch_tensors_basics.ipynb'-jupyter notebook.
That notebook tests your knowledge about the basic torch.Tensor-functions.
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
import torch_tensors_basics


class TestBasics(unittest.TestCase):
    """
    The class contains all test cases for task "2 - Basic Torch".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the test parameters
        self.shape = (2, 3, 4)
        self.lst = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
        self.np_array = np.array([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
        self.tensor = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        self.tensor_slicing = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])

    def test_creation_of_tensor_with_specified_shape(self):
        """ Test creation of tensor with specified shape. """

        student_version = torch_tensors_basics.create_tensor_with_specified_shape(self.shape)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_basics.pt")["t1"]

        self.assertEqual(student_version.size(), expected_version.size(), "Shape of tensor is not correct!")

    def test_creation_of_tensor_from_list(self):
        """ Test creation of tensor from a list. """

        student_version = torch_tensors_basics.create_tensor_from_list(self.lst)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_basics.pt")["t2"]

        self.assertEqual(student_version.size(), expected_version.size(), "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

    def test_creation_of_tensor_from_numpy_array(self):
        """ Test creation of tensor from a numpy array. """

        student_version = torch_tensors_basics.create_tensor_from_numpy_array(self.np_array)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_basics.pt")["t3"]

        self.assertEqual(student_version.size(), expected_version.size(), "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

    def test_creation_of_random_tensor(self):
        """ Test creation of a random tensor with a specified shape. """

        student_version = torch_tensors_basics.create_random_tensor(self.shape)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_basics.pt")["t4"]

        self.assertEqual(student_version.size(), expected_version.size(), "Shape of tensor is not correct!")

    def test_creation_of_zero_tensor(self):
        """ Test creation of a tensor with a specified shape filled with zeros. """

        student_version = torch_tensors_basics.create_tensor_filled_with_zeros(self.shape)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_basics.pt")["t5"]

        self.assertEqual(student_version.size(), expected_version.size(), "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

    def test_creation_of_one_tensor(self):
        """ Test creation of a tensor with a specified shape filled with ones. """

        student_version = torch_tensors_basics.create_tensor_filled_with_ones(self.shape)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_basics.pt")["t6"]

        self.assertEqual(student_version.size(), expected_version.size(), "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

    def test_creation_of_one_diagonal_tensor(self):
        """ Test creation of 2D-tensor with a specified number of rows and columns for which its diagonal is filled
        with ones. """

        student_version = torch_tensors_basics.create_tensor_with_the_diagonal_filled_with_ones((self.shape[0],
                                                                                                 self.shape[1]))
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_basics.pt")["t7"]

        self.assertEqual(student_version.size(), expected_version.size(), "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

    def test_extraction_of_tensor_type(self):
        """ Test extracting the type of a tensor. """

        student_version = torch_tensors_basics.extract_the_type_of_a_tensor(self.tensor)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_basics.pt")["t8"]

        self.assertEqual(student_version, expected_version, "Type of tensor is not correct!")

    def test_extraction_of_tensor_data_type(self):
        """ Test extracting the data type of a tensor. """

        student_version = torch_tensors_basics.extract_the_data_type_of_a_tensor(self.tensor)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_basics.pt")["t9"]

        self.assertEqual(student_version, expected_version, "Data type of tensor is not correct!")

    def test_extraction_of_tensor_dimension(self):
        """ Test extracting the dimension of a tensor. """

        student_version = torch_tensors_basics.extract_the_dimensions_of_a_tensor(self.tensor)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_basics.pt")["t10"]

        self.assertEqual(student_version, expected_version, "Dimension of tensor is not correct!")

    def test_extraction_of_tensor_shape(self):
        """ Test extracting the shape of a tensor. """

        student_version = torch_tensors_basics.extract_the_shape_of_a_tensor(self.tensor)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_basics.pt")["t11"]

        self.assertEqual(student_version, expected_version, "Shape of tensor is not correct!")

    def test_extraction_of_tensor_number_elements(self):
        """ Test extracting the number of elements of a tensor. """

        student_version = torch_tensors_basics.extract_the_number_of_elements_of_a_tensor(self.tensor)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_basics.pt")["t12"]

        self.assertEqual(student_version, expected_version, "Number of elements of tensor is not correct!")

    def test_slicing_01(self):
        """ Test slicing of a tensor. """

        student_version = torch_tensors_basics.slicing_01(self.tensor_slicing)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_basics.pt")["t13"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

    def test_slicing_02(self):
        """ Test different slicing of a tensor. """

        student_version = torch_tensors_basics.slicing_02(self.tensor_slicing)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_basics.pt")["t14"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

    def test_slicing_03(self):
        """ Test different slicing of a tensor. """

        student_version = torch_tensors_basics.slicing_03(self.tensor_slicing)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_basics.pt")["t15"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")

    def test_slicing_04(self):
        """ Test different slicing of a tensor. """

        student_version = torch_tensors_basics.slicing_04(self.tensor_slicing)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_basics.pt")["t16"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of tensor is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of tensor is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Tensor is not correct!")


class TestTensorOperations(unittest.TestCase):
    """
    The class contains all test cases for task "3 - Tensor Operations".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the test parameters
        self.t1 = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
        self.t2 = torch.tensor([[41, 42, 43], [51, 52, 53], [61, 62, 63]])
        self.scalar = 2.0
        self.new_shape = (9,)
        self.axis = 0

    def test_summation_of_tensors(self):
        """ Test summation of two tensors. """

        student_version = torch_tensors_basics.tensor_summation(self.t1, self.t2)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_operations.pt")["t1"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

    def test_scalar_multiplication(self):
        """ Test multiplying a scalar with a tensor. """

        student_version = torch_tensors_basics.scalar_multiplication(self.t1, self.scalar)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_operations.pt")["t2"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

    def test_hadamard_product(self):
        """ Test calculating the hadamard product of two tensors. """

        student_version = torch_tensors_basics.hadamard_product(self.t1, self.t2)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_operations.pt")["t3"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

    def test_matrix_product(self):
        """ Test calculating the matrix product of two tensors. """

        student_version = torch_tensors_basics.matrix_product(self.t1, self.t2)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_operations.pt")["t4"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

    def test_dot_product(self):
        """ Test calculating the dot product of two tensors. """

        student_version = torch_tensors_basics.dot_product(self.t1.flatten(), self.t2.flatten())
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_operations.pt")["t5"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

    def test_square_tensor(self):
        """ Test squaring all elements of a tensor. """

        student_version = torch_tensors_basics.square_tensor(self.t1)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_operations.pt")["t6"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

    def test_minimum(self):
        """ Test finding the minimum of a tensor. """

        student_version = torch_tensors_basics.minimum(self.t1)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_operations.pt")["t7"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

    def test_maximum(self):
        """ Test finding the maximum of a tensor. """

        student_version = torch_tensors_basics.maximum(self.t1)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_operations.pt")["t8"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

    def test_mean(self):
        """ Test finding the mean of a tensor. """

        student_version = torch_tensors_basics.mean(self.t1)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_operations.pt")["t9"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

    def test_standard_deviation(self):
        """ Test finding the standard deviation of a tensor. """

        student_version = torch_tensors_basics.standard_deviation(self.t1)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_operations.pt")["t10"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

    def test_cast_to_list(self):
        """ Test casting a tensor to a Python list. """

        student_version = torch_tensors_basics.cast_to_list(self.t1)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_operations.pt")["t11"]

        self.assertEqual(type(student_version), type(expected_version), "Type of solution is not correct!")
        self.assertEqual(len(student_version), len(expected_version), "Shape of solution is not correct!")
        self.assertTrue(student_version == expected_version, "Solution is not correct!")

    def test_cast_to_numpy(self):
        """ Test casting a tensor to a Numpy array. """

        student_version = torch_tensors_basics.cast_to_numpy(self.t1)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_operations.pt")["t12"]

        self.assertEqual(type(student_version), type(expected_version), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(np.array_equal(student_version, expected_version), "Solution is not correct!")

    def test_cast_to_double(self):
        """ Test casting an integer tensor to a double tensor. """

        student_version = torch_tensors_basics.cast_to_double(self.t1)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_operations.pt")["t13"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

    def test_reshape(self):
        """ Test reshaping a tensor to a new shape. """

        student_version = torch_tensors_basics.reshape(self.t1, self.new_shape)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_operations.pt")["t14"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

    def test_swap(self):
        """ Test swapping the first axis of a 3D-tensor with its last one. """

        student_version = torch_tensors_basics.swap(torch.unsqueeze(self.t1, dim=0))
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_operations.pt")["t15"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

    def test_trans(self):
        """ Test transposing a tensor. """

        student_version = torch_tensors_basics.trans(self.t1)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_operations.pt")["t16"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

    def test_pave(self):
        """ Test paving an nD-tensor to 1D. """

        student_version = torch_tensors_basics.pave(self.t1)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_operations.pt")["t17"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

    def test_concat1(self):
        """ Test concatenating two tensors along a specified axis. """

        t1 = torch.tensor([[11, 12], [21, 22], [31, 32]])
        t2 = torch.tensor([[41, 42], [51, 52], [61, 62]])

        student_version = torch_tensors_basics.concat(t1, t2, self.axis)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_operations.pt")["t18"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")

    def test_concat2(self):
        """ Test concatenating another two tensors along a specified axis. """

        t1 = torch.tensor([[11, 12], [21, 22], [31, 32]])
        t2 = torch.tensor([[41, 42, 43], [51, 52, 53]])

        student_version = torch_tensors_basics.concat(t1, t2, self.axis)
        expected_version = torch.load("../data/torch_tensors_basics/reference_tensor_operations.pt")["t19"]

        self.assertEqual(student_version.type(), expected_version.type(), "Type of solution is not correct!")
        self.assertEqual(student_version.shape, expected_version.shape, "Shape of solution is not correct!")
        self.assertTrue(torch.allclose(student_version, expected_version, atol=0.0001), "Solution is not correct!")


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