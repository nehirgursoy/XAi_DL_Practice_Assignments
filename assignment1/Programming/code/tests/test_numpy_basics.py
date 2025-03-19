#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
This script was written as a test module for the 'numpy_basics.ipynb'-jupyter notebook.
That notebook introduces the basic functions and processes of the free numpy package.
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
import unittest
import argparse

# Import own files
import nbimporter  # Necessary to be able to use equations of ipynb-notebooks in python-files
import numpy_basics


class TestArrayProperties(unittest.TestCase):
    """
    The class contains all test cases for task 1.1 - General Array Properties.
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the tests
        self.np_a1 = np.array([0, 1, 2, 3, 4, 5])
        self.np_a2 = np.array([[0, 1], [1, 0]])
        self.np_a3 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    def test_shape(self):
        """ Test shape of arrays. """

        student_shape1 = numpy_basics.extract_array_dimensions_01(self.np_a1)
        student_shape2 = numpy_basics.extract_array_dimensions_01(self.np_a2)
        student_shape3 = numpy_basics.extract_array_dimensions_01(self.np_a3)
        expected_shape1 = (6, )
        expected_shape2 = (2, 2)
        expected_shape3 = (3, 3)

        self.assertEqual(expected_shape1, student_shape1)
        self.assertEqual(expected_shape2, student_shape2)
        self.assertEqual(expected_shape3, student_shape3)

    def test_size(self):
        """ Test size of arrays. """

        student_size1 = numpy_basics.extract_array_dimensions_02(self.np_a1)
        student_size2 = numpy_basics.extract_array_dimensions_02(self.np_a2)
        student_size3 = numpy_basics.extract_array_dimensions_02(self.np_a3)
        expected_size1 = 6
        expected_size2 = 4
        expected_size3 = 9

        self.assertEqual(expected_size1, student_size1)
        self.assertEqual(expected_size2, student_size2)
        self.assertEqual(expected_size3, student_size3)


class TestArrayCreation(unittest.TestCase):
    """
    The class contains all test cases for task 1.2 - Creating Arrays.
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the tests
        self.zeros_shape1 = (1, )
        self.zeros_shape2 = (2, 2)
        self.zeros_shape3 = (3, 5, 4, 1)

        self.ones_shape1 = (1, )
        self.ones_shape2 = (2, 2)
        self.ones_shape3 = (3, 5, 4, 1)

        self.array_like1 = np.array([0, 1, 2, 3, 4, 5])
        self.array_like2 = np.array([[0, 1], [1, 0]])
        self.array_like3 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

        self.range1 = [0, 11, 1]
        self.range2 = [-5, 5, 2]
        self.range3 = [100, 1000, 10]

        self.elements1 = [0, 11, 11]
        self.elements2 = [-5, 5, 100]
        self.elements3 = [0, 1, 1]

    def test_zeros(self):
        """ Test creation of array filled only with zeros. """

        student_version1 = numpy_basics.create_numpy_arrays_01(self.zeros_shape1)
        student_version2 = numpy_basics.create_numpy_arrays_01(self.zeros_shape2)
        student_version3 = numpy_basics.create_numpy_arrays_01(self.zeros_shape3)

        # Test 1
        self.assertEqual(student_version1.shape, self.zeros_shape1)
        self.assertEqual(np.sum(student_version1), 0)

        # Test 2
        self.assertEqual(student_version2.shape, self.zeros_shape2)
        self.assertEqual(np.sum(student_version2), 0)

        # Test 3
        self.assertEqual(student_version3.shape, self.zeros_shape3)
        self.assertEqual(np.sum(student_version3), 0)

    def test_ones(self):
        """ Test creation of array filled only with ones. """

        student_version1 = numpy_basics.create_numpy_arrays_02(self.ones_shape1)
        student_version2 = numpy_basics.create_numpy_arrays_02(self.ones_shape2)
        student_version3 = numpy_basics.create_numpy_arrays_02(self.ones_shape3)

        # Test 1
        self.assertEqual(student_version1.shape, self.ones_shape1)
        self.assertEqual(np.max(student_version1), 1)
        self.assertEqual(np.min(student_version1), 1)

        # Test 2
        self.assertEqual(student_version2.shape, self.ones_shape2)
        self.assertEqual(np.max(student_version2), 1)
        self.assertEqual(np.min(student_version2), 1)

        # Test 3
        self.assertEqual(student_version3.shape, self.ones_shape3)
        self.assertEqual(np.max(student_version3), 1)
        self.assertEqual(np.min(student_version3), 1)

    def test_array_like(self):
        """ Test creation of array which entries shall be replaced with zeros. """

        student_version1 = numpy_basics.create_numpy_arrays_03(self.array_like1)
        student_version2 = numpy_basics.create_numpy_arrays_03(self.array_like2)
        student_version3 = numpy_basics.create_numpy_arrays_03(self.array_like3)

        # Test 1
        self.assertEqual(student_version1.shape, self.array_like1.shape)
        self.assertEqual(np.sum(student_version1), 0)

        # Test 2
        self.assertEqual(student_version2.shape, self.array_like2.shape)
        self.assertEqual(np.sum(student_version2), 0)

        # Test 3
        self.assertEqual(student_version3.shape, self.array_like3.shape)
        self.assertEqual(np.sum(student_version3), 0)

    def test_range(self):
        """ Test creation of ranged array. """

        student_version1 = numpy_basics.create_numpy_arrays_04(self.range1)
        student_version2 = numpy_basics.create_numpy_arrays_04(self.range2)
        student_version3 = numpy_basics.create_numpy_arrays_04(self.range3)

        # Test 1
        self.assertEqual(student_version1.shape, (11, ))
        self.assertEqual(np.sum(student_version1), 55)

        # Test 2
        self.assertEqual(student_version2.shape, (5, ))
        self.assertEqual(np.sum(student_version2), -5)

        # Test 3
        self.assertEqual(student_version3.shape, (90, ))
        self.assertEqual(np.sum(student_version3), 49050)

    def test_elements(self):
        """ Test creation of array with specified amount of elements. """

        student_version1 = numpy_basics.create_numpy_arrays_05(self.elements1)
        student_version2 = numpy_basics.create_numpy_arrays_05(self.elements2)
        student_version3 = numpy_basics.create_numpy_arrays_05(self.elements3)

        # Test 1
        self.assertEqual(student_version1.shape, (11, ))
        self.assertAlmostEqual(float(np.sum(student_version1)), 60.5, None, "Values are not almost equal!", 0.0001)

        # Test 2
        self.assertEqual(student_version2.shape, (100, ))
        self.assertAlmostEqual(float(np.sum(student_version2)), 0.0, None, "Values are not almost equal!", 0.0001)

        # Test 3
        self.assertEqual(student_version3.shape, (1, ))
        self.assertAlmostEqual(float(np.sum(student_version3)), 0.0, None, "Values are not almost equal!", 0.0001)


class TestArrayHandling(unittest.TestCase):
    """
    The class contains all test cases for task 1.3 - Array Handling.
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the tests
        # Reshape array
        self.reshape_array1 = np.array([0, 1, 2, 3, 4, 5])
        self.reshape_array2 = np.array([[0, 1], [1, 0]])
        self.reshape_array3 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

        self.reshaped_shape1 = (2, 3)
        self.reshaped_shape2 = (1, 4)
        self.reshaped_shape3 = (1, 1, 9)

        # Vertical and horizontal stack
        self.stack1 = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
        self.stack2 = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9], [10, 11, 12]])]
        self.stack3 = [np.array([[1], [2], [3]]), np.array([[4], [5], [6]]), np.array([[7], [8], [9]])]

        # Concatenating
        self.concat_arrays1 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        self.concat_arrays2 = [np.array([[1, 2, 3], [3, 4, 5]]), np.array([[6, 7, 8, 9], [10, 11, 12, 13]])]
        self.concat_arrays3 = [np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]], [[5, 5], [6, 6]]]),
                               np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]])]

        self.concat_axis1 = 0
        self.concat_axis2 = 1
        self.concat_axis3 = 2

        # Repeating
        self.repeat_array1 = np.array([0, 1, 2, 3, 4, 5])
        self.repeat_array2 = np.array([[0, 1], [1, 0]])
        self.repeat_array3 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

        self.repeat_times1 = (2, )
        self.repeat_times2 = (1, 2)
        self.repeat_times3 = (2, 2)

        # Transposing
        self.transpose_array1 = np.array([0, 1, 2, 3, 4, 5])
        self.transpose_array2 = np.array([[0, 1], [1, 0]])
        self.transpose_array3 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    def test_reshape(self):
        """ Test reshaping arrays. """

        student_version1 = numpy_basics.handle_numpy_arrays_01([self.reshape_array1, self.reshaped_shape1])
        student_version2 = numpy_basics.handle_numpy_arrays_01([self.reshape_array2, self.reshaped_shape2])
        student_version3 = numpy_basics.handle_numpy_arrays_01([self.reshape_array3, self.reshaped_shape3])

        # Test 1
        self.assertEqual(student_version1.shape, self.reshaped_shape1)
        self.assertEqual(np.min(student_version1), 0)
        self.assertEqual(np.max(student_version1), 5)
        self.assertEqual(np.sum(student_version1), 15)
        self.assertTrue(np.allclose(student_version1[0], np.array([0, 1, 2])))

        # Test 2
        self.assertEqual(student_version2.shape, self.reshaped_shape2)
        self.assertEqual(np.min(student_version2), 0)
        self.assertEqual(np.max(student_version2), 1)
        self.assertEqual(np.sum(student_version2), 2)
        self.assertTrue(np.allclose(student_version2[0], np.array([0, 1, 1, 0])))

        # Test 3
        self.assertEqual(student_version3.shape, self.reshaped_shape3)
        self.assertEqual(np.min(student_version3), -1)
        self.assertEqual(np.max(student_version3), 1)
        self.assertEqual(np.sum(student_version3), 0)
        self.assertTrue(np.allclose(student_version3[0], np.array([[1,  1,  1,  0,  0,  0, -1, -1, -1]])))

    def test_vstack(self):
        """ Test vertically stacking arrays. """

        student_version1 = numpy_basics.handle_numpy_arrays_02(self.stack1)
        student_version2 = numpy_basics.handle_numpy_arrays_02(self.stack2)
        student_version3 = numpy_basics.handle_numpy_arrays_02(self.stack3)

        # Test 1
        self.assertEqual(student_version1.shape, (3, 3))
        self.assertEqual(np.min(student_version1), 1)
        self.assertEqual(np.max(student_version1), 9)
        self.assertEqual(np.sum(student_version1), 45)
        self.assertTrue(np.allclose(student_version1[0], np.array([1, 2, 3])))

        # Test 2
        self.assertEqual(student_version2.shape, (4, 3))
        self.assertEqual(np.min(student_version2), 1)
        self.assertEqual(np.max(student_version2), 12)
        self.assertEqual(np.sum(student_version2), 78)
        self.assertTrue(np.allclose(student_version2[0], np.array([1, 2, 3])))
        self.assertTrue(np.allclose(student_version2[2], np.array([7, 8, 9])))

        # Test 3
        self.assertEqual(student_version3.shape, (9, 1))
        self.assertEqual(np.min(student_version3), 1)
        self.assertEqual(np.max(student_version3), 9)
        self.assertEqual(np.sum(student_version3), 45)
        self.assertTrue(np.allclose(student_version3[0], np.array([1])))
        self.assertTrue(np.allclose(student_version3[4], np.array([5])))

    def test_hstack(self):
        """ Test horizontally stacking arrays. """

        student_version1 = numpy_basics.handle_numpy_arrays_03(self.stack1)
        student_version2 = numpy_basics.handle_numpy_arrays_03(self.stack2)
        student_version3 = numpy_basics.handle_numpy_arrays_03(self.stack3)

        # Test 1
        self.assertEqual(student_version1.shape, (9,))
        self.assertEqual(np.min(student_version1), 1)
        self.assertEqual(np.max(student_version1), 9)
        self.assertEqual(np.sum(student_version1), 45)
        self.assertTrue(np.allclose(student_version1[0], np.array([1])))

        # Test 2
        self.assertEqual(student_version2.shape, (2, 6))
        self.assertEqual(np.min(student_version2), 1)
        self.assertEqual(np.max(student_version2), 12)
        self.assertEqual(np.sum(student_version2), 78)
        self.assertTrue(np.allclose(student_version2[0], np.array([1, 2, 3, 7, 8, 9])))

        # Test 3
        self.assertEqual(student_version3.shape, (3, 3))
        self.assertEqual(np.min(student_version3), 1)
        self.assertEqual(np.max(student_version3), 9)
        self.assertEqual(np.sum(student_version3), 45)
        self.assertTrue(np.allclose(student_version3[0], np.array([1, 4, 7])))

    def test_concatenation(self):
        """ Test concatenating of arrays. """

        student_version1 = numpy_basics.handle_numpy_arrays_04([*self.concat_arrays1, self.concat_axis1])
        student_version2 = numpy_basics.handle_numpy_arrays_04([*self.concat_arrays2, self.concat_axis2])
        student_version3 = numpy_basics.handle_numpy_arrays_04([*self.concat_arrays3, self.concat_axis3])

        # Test 1
        self.assertEqual(student_version1.shape, (6,))
        self.assertEqual(np.min(student_version1), 1)
        self.assertEqual(np.max(student_version1), 6)
        self.assertEqual(np.sum(student_version1), 21)
        self.assertTrue(np.allclose(student_version1[0], np.array([1])))

        # Test 2
        self.assertEqual(student_version2.shape, (2, 7))
        self.assertEqual(np.min(student_version2), 1)
        self.assertEqual(np.max(student_version2), 13)
        self.assertEqual(np.sum(student_version2), 94)
        self.assertTrue(np.allclose(student_version2[0], np.array([1, 2, 3, 6, 7, 8, 9])))

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2, 5))
        self.assertEqual(np.min(student_version3), 1)
        self.assertEqual(np.max(student_version3), 6)
        self.assertEqual(np.sum(student_version3), 105)
        self.assertTrue(np.allclose(student_version3[0], np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])))

    def test_repeating(self):
        """ Test repeating arrays. """

        student_version1 = numpy_basics.handle_numpy_arrays_05([self.repeat_array1, self.repeat_times1])
        student_version2 = numpy_basics.handle_numpy_arrays_05([self.repeat_array2, self.repeat_times2])
        student_version3 = numpy_basics.handle_numpy_arrays_05([self.repeat_array3, self.repeat_times3])

        # Test 1
        self.assertEqual(student_version1.shape, (12,))
        self.assertEqual(np.min(student_version1), 0)
        self.assertEqual(np.max(student_version1), 5)
        self.assertEqual(np.sum(student_version1), 30)
        self.assertTrue(np.allclose(student_version1[0], 0))

        # Test 2
        self.assertEqual(student_version2.shape, (2, 4))
        self.assertEqual(np.min(student_version2), 0)
        self.assertEqual(np.max(student_version2), 1)
        self.assertEqual(np.sum(student_version2), 4)
        self.assertTrue(np.allclose(student_version2[0], np.array([0, 1, 0, 1])))

        # Test 3
        self.assertEqual(student_version3.shape, (6, 6))
        self.assertEqual(np.min(student_version3), -1)
        self.assertEqual(np.max(student_version3), 1)
        self.assertEqual(np.sum(student_version3), 0)
        self.assertTrue(np.allclose(student_version3[0], np.array([1, 1, 1, 1, 1, 1])))

    def test_transposing(self):
        """ Test transposing arrays. """

        student_version1 = numpy_basics.handle_numpy_arrays_06(self.transpose_array1)
        student_version2 = numpy_basics.handle_numpy_arrays_06(self.transpose_array2)
        student_version3 = numpy_basics.handle_numpy_arrays_06(self.transpose_array3)

        # Test 1
        self.assertEqual(student_version1.shape, (6,))
        self.assertEqual(np.min(student_version1), 0)
        self.assertEqual(np.max(student_version1), 5)
        self.assertEqual(np.sum(student_version1), 15)
        self.assertTrue(np.allclose(student_version1[0], 0))

        # Test 2
        self.assertEqual(student_version2.shape, (2, 2))
        self.assertEqual(np.min(student_version2), 0)
        self.assertEqual(np.max(student_version2), 1)
        self.assertEqual(np.sum(student_version2), 2)
        self.assertTrue(np.allclose(student_version2[0], np.array([0, 1])))

        # Test 3
        self.assertEqual(student_version3.shape, (3, 3))
        self.assertEqual(np.min(student_version3), -1)
        self.assertEqual(np.max(student_version3), 1)
        self.assertEqual(np.sum(student_version3), 0)
        self.assertTrue(np.allclose(student_version3[0], np.array([1, 0, -1])))


class TestArrayCalculations(unittest.TestCase):
    """
    The class contains all test cases for task 1.4 - Calculating with Arrays.
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the tests
        self.array11 = np.array([1, 2, 3])
        self.array12 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        self.array13 = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]], [[5, 5], [6, 6]]])

        self.array21 = np.array([6, 7, 8])
        self.array22 = np.array([[5, 5, 5], [1, 2, 3], [18, 19, 20]])
        self.array23 = np.array([[[12, 12], [11, 11]], [[10, 10], [9, 9]], [[8, 8], [7, 7]]])

        self.sum_axis1 = 0
        self.sum_axis2 = 1
        self.sum_axis3 = 2

        self.threshold1 = 3
        self.threshold2 = 0
        self.threshold3 = 4

    def test_adding(self):
        """ Test adding arrays element-wise. """

        student_version1 = numpy_basics.calculations_with_numpy_arrays_01([self.array11, self.array21])
        student_version2 = numpy_basics.calculations_with_numpy_arrays_01([self.array12, self.array22])
        student_version3 = numpy_basics.calculations_with_numpy_arrays_01([self.array13, self.array23])

        # Test 1
        self.assertEqual(student_version1.shape, (3,))
        self.assertEqual(np.sum(student_version1), 27)

        # Test 2
        self.assertEqual(student_version2.shape, (3, 3))
        self.assertEqual(np.sum(student_version2), 78)

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2, 2))
        self.assertEqual(np.sum(student_version3), 156)

    def test_dot_product(self):
        """ Test dot product of two arrays. """

        student_version1 = numpy_basics.calculations_with_numpy_arrays_02([self.array11, self.array21])
        student_version2 = numpy_basics.calculations_with_numpy_arrays_02([self.array12, self.array22])
        student_version3 = numpy_basics.calculations_with_numpy_arrays_02([self.array13, self.array23])

        # Test 1
        self.assertEqual(student_version1.shape, ())
        self.assertEqual(np.sum(student_version1), 44)

        # Test 2
        self.assertEqual(student_version2.shape, (3, 3))
        self.assertEqual(np.sum(student_version2), 0)

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2, 3, 2))
        self.assertEqual(np.sum(student_version3), 2394)

    def test_multiply(self):
        """ Test multiplying arrays elment-wise. """

        student_version1 = numpy_basics.calculations_with_numpy_arrays_03([self.array11, self.array21])
        student_version2 = numpy_basics.calculations_with_numpy_arrays_03([self.array12, self.array22])
        student_version3 = numpy_basics.calculations_with_numpy_arrays_03([self.array13, self.array23])

        # Test 1
        self.assertEqual(student_version1.shape, (3, ))
        self.assertEqual(np.sum(student_version1), 44)

        # Test 2
        self.assertEqual(student_version2.shape, (3, 3))
        self.assertEqual(np.sum(student_version2), -42)

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2, 2))
        self.assertEqual(np.sum(student_version3), 364)

    def test_sum(self):
        """ Test summing arrays along a specified axis. """

        student_version1 = numpy_basics.calculations_with_numpy_arrays_04([self.array11, self.sum_axis1])
        student_version2 = numpy_basics.calculations_with_numpy_arrays_04([self.array12, self.sum_axis2])
        student_version3 = numpy_basics.calculations_with_numpy_arrays_04([self.array13, self.sum_axis3])

        # Test 1
        self.assertEqual(student_version1.shape, ())
        self.assertEqual(np.sum(student_version1), 6)

        # Test 2
        self.assertEqual(student_version2.shape, (3, ))
        self.assertEqual(np.sum(student_version2), 0)

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2))
        self.assertEqual(np.sum(student_version3), 42)

    def test_grad(self):
        """ Test gradient of array. """

        student_version1 = numpy_basics.calculations_with_numpy_arrays_05(self.array11)
        student_version2 = numpy_basics.calculations_with_numpy_arrays_05(self.array12)
        student_version3 = numpy_basics.calculations_with_numpy_arrays_05(self.array13)

        # Test 1
        self.assertEqual(student_version1.shape, (3,))
        self.assertEqual(np.sum(student_version1), 3.0)

        # Test 2
        self.assertEqual(student_version2[0].shape, (3, 3))
        self.assertEqual(np.sum(student_version2), -9.0)

        # Test 3
        self.assertEqual(student_version3[0].shape, (3, 2, 2))
        self.assertEqual(np.sum(student_version3), 36.0)

    def test_cross(self):
        """ Test cross product of arrays. """

        student_version1 = numpy_basics.calculations_with_numpy_arrays_06([self.array11, self.array21])
        student_version2 = numpy_basics.calculations_with_numpy_arrays_06([self.array12, self.array22])
        student_version3 = numpy_basics.calculations_with_numpy_arrays_06([self.array13, self.array23])

        # Test 1
        self.assertEqual(student_version1.shape, (3,))
        self.assertEqual(np.sum(student_version1), 0)

        # Test 2
        self.assertEqual(student_version2.shape, (3, 3))
        self.assertEqual(np.sum(student_version2), 0)

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2))
        self.assertEqual(np.sum(student_version3), 0)

    def test_square_root(self):
        """ Test element-wise square root of array. """

        student_version1 = numpy_basics.calculations_with_numpy_arrays_07(self.array11)
        student_version2 = numpy_basics.calculations_with_numpy_arrays_07(self.array22)
        student_version3 = numpy_basics.calculations_with_numpy_arrays_07(self.array13)

        # Test 1
        self.assertEqual(student_version1.shape, (3,))
        self.assertAlmostEqual(float(np.sum(student_version1)), 4.146264369941973, None, "Values are not almost equal!",
                               0.0001)

        # Test 2
        self.assertEqual(student_version2.shape, (3, 3))
        self.assertAlmostEqual(float(np.sum(student_version2)), 23.92814388810088, None, "Values are not almost equal!",
                               0.0001)

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2, 2))
        self.assertAlmostEqual(float(np.sum(student_version3)), 21.663644180449882, None, "Values are not almost equal!",
                               0.0001)

    def test_masking(self):
        """ Test masking of array. """

        student_version1 = numpy_basics.calculations_with_numpy_arrays_08([self.array11, self.threshold1])
        student_version2 = numpy_basics.calculations_with_numpy_arrays_08([self.array12, self.threshold2])
        student_version3 = numpy_basics.calculations_with_numpy_arrays_08([self.array13, self.threshold3])

        # Test 1
        self.assertEqual(student_version1.shape, (2,))
        self.assertEqual(np.sum(student_version1), 3)

        # Test 2
        self.assertEqual(student_version2.shape, (3,))
        self.assertEqual(np.sum(student_version2), -3)

        # Test 3
        self.assertEqual(student_version3.shape, (6, ))
        self.assertEqual(np.sum(student_version3), 12)


class TestArrayStatistics(unittest.TestCase):
    """
    The class contains all test cases for task 1.5 - Statistics with Arrays.
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the tests
        self.array1 = np.array([1, 2, 3])
        self.array2 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        self.array3 = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]], [[5, 5], [6, 6]]])

        self.axis1 = None
        self.axis2 = 1
        self.axis3 = 2

    def test_min_value(self):
        """ Test extracting the minimum value of an array. """

        student_version1 = numpy_basics.statistics_with_numpy_arrays_01(self.array1, self.axis1)
        student_version2 = numpy_basics.statistics_with_numpy_arrays_01(self.array2, self.axis2)
        student_version3 = numpy_basics.statistics_with_numpy_arrays_01(self.array3, self.axis3)

        # Test 1
        self.assertEqual(student_version1.shape, ())
        self.assertEqual(np.sum(student_version1), 1)

        # Test 2
        self.assertEqual(student_version2.shape, (3,))
        self.assertEqual(np.sum(student_version2), 0)

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2))
        self.assertEqual(np.sum(student_version3), 21)

    def test_max_value(self):
        """ Test extracting the maximum value of an array. """

        student_version1 = numpy_basics.statistics_with_numpy_arrays_02(self.array1, self.axis1)
        student_version2 = numpy_basics.statistics_with_numpy_arrays_02(self.array2, self.axis2)
        student_version3 = numpy_basics.statistics_with_numpy_arrays_02(self.array3, self.axis3)

        # Test 1
        self.assertEqual(student_version1.shape, ())
        self.assertEqual(np.sum(student_version1), 3)

        # Test 2
        self.assertEqual(student_version2.shape, (3,))
        self.assertEqual(np.sum(student_version2), 0)

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2))
        self.assertEqual(np.sum(student_version3), 21)

    def test_min_index(self):
        """ Test extracting the index of the minimum value of an array. """

        student_version1 = numpy_basics.statistics_with_numpy_arrays_03(self.array1, self.axis1)
        student_version2 = numpy_basics.statistics_with_numpy_arrays_03(self.array2, self.axis2)
        student_version3 = numpy_basics.statistics_with_numpy_arrays_03(self.array3, self.axis3)

        # Test 1
        self.assertEqual(student_version1.shape, ())
        self.assertEqual(np.sum(student_version1), 0)

        # Test 2
        self.assertEqual(student_version2.shape, (3,))
        self.assertEqual(np.sum(student_version2), 0)

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2))
        self.assertEqual(np.sum(student_version3), 0)

    def test_max_index(self):
        """ Test extracting the index of the maximum value of an array. """

        student_version1 = numpy_basics.statistics_with_numpy_arrays_04(self.array1, self.axis1)
        student_version2 = numpy_basics.statistics_with_numpy_arrays_04(self.array2, self.axis2)
        student_version3 = numpy_basics.statistics_with_numpy_arrays_04(self.array3, self.axis3)

        # Test 1
        self.assertEqual(student_version1.shape, ())
        self.assertEqual(np.sum(student_version1), 2)

        # Test 2
        self.assertEqual(student_version2.shape, (3,))
        self.assertEqual(np.sum(student_version2), 0)

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2))
        self.assertEqual(np.sum(student_version3), 0)

    def test_mean(self):
        """ Test calculating the mean of an array along a given axis. """

        student_version1 = numpy_basics.statistics_with_numpy_arrays_05(self.array1, self.axis1)
        student_version2 = numpy_basics.statistics_with_numpy_arrays_05(self.array2, self.axis2)
        student_version3 = numpy_basics.statistics_with_numpy_arrays_05(self.array3, self.axis3)

        # Test 1
        self.assertEqual(student_version1.shape, ())
        self.assertEqual(np.sum(student_version1), 2.0)

        # Test 2
        self.assertEqual(student_version2.shape, (3,))
        self.assertEqual(np.sum(student_version2), 0)

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2))
        self.assertEqual(np.sum(student_version3), 21)

    def test_std(self):
        """ Test calculating the standard deviation of an array along a given axis. """

        student_version1 = numpy_basics.statistics_with_numpy_arrays_06(self.array1, self.axis1)
        student_version2 = numpy_basics.statistics_with_numpy_arrays_06(self.array2, self.axis2)
        student_version3 = numpy_basics.statistics_with_numpy_arrays_06(self.array3, self.axis3)

        # Test 1
        self.assertEqual(student_version1.shape, ())
        self.assertAlmostEqual(float(np.sum(student_version1)), 0.8164, None, "Values are not almost equal!", 0.0001)

        # Test 2
        self.assertEqual(student_version2.shape, (3,))
        self.assertEqual(np.sum(student_version2), 0)

        # Test 3
        self.assertEqual(student_version3.shape, (3, 2))
        self.assertEqual(np.sum(student_version3), 0)


class TestArraySlicing1D(unittest.TestCase):
    """
    The class contains the first test cases for task 1.6 - Array Slicing.
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the tests
        self.array1 = None
        self.array2 = np.arange(1, 11)
        self.array3 = np.array([1, 5, 6, 9, 13, 4, -5, 17, -3, 2, 3, 7, -5])

    def test_first_element(self):
        """ Test extracting the first element of an array. """

        student_version1 = numpy_basics.slicing_of_numpy_arrays_1d_01(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_1d_01(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_1d_01(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertEqual(student_version2.shape, ())
        self.assertEqual(np.sum(student_version2), 1)

        # Test 3
        self.assertEqual(student_version3.shape, ())
        self.assertEqual(np.sum(student_version3), 1)

    def test_last_element(self):
        """ Test extracting the last element of an array. """

        student_version1 = numpy_basics.slicing_of_numpy_arrays_1d_02(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_1d_02(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_1d_02(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertEqual(student_version2.shape, ())
        self.assertEqual(np.sum(student_version2), 10)

        # Test 3
        self.assertEqual(student_version3.shape, ())
        self.assertEqual(np.sum(student_version3), -5)

    def test_first_half(self):
        """ Test extracting the first half of an array. """

        student_version1 = numpy_basics.slicing_of_numpy_arrays_1d_03(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_1d_03(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_1d_03(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (5,))
        self.assertEqual(np.sum(student_version2), 15)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (6,))
        self.assertEqual(np.sum(student_version3), 38)

    def test_second_half(self):
        """ Test extracting the second half of an array. """

        student_version1 = numpy_basics.slicing_of_numpy_arrays_1d_04(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_1d_04(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_1d_04(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (5,))
        self.assertEqual(np.sum(student_version2), 40)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (7,))
        self.assertEqual(np.sum(student_version3), 16)

    def test_elements_at_odd_positions(self):
        """ Test extracting all elements at an odd position within an array. """

        student_version1 = numpy_basics.slicing_of_numpy_arrays_1d_05(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_1d_05(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_1d_05(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (5,))
        self.assertEqual(np.sum(student_version2), 25)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (7,))
        self.assertEqual(np.sum(student_version3), 10)

    def test_elements_at_even_positions(self):
        """ Test extracting all elements at an even position within an array. """

        student_version1 = numpy_basics.slicing_of_numpy_arrays_1d_06(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_1d_06(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_1d_06(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (5,))
        self.assertEqual(np.sum(student_version2), 30)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (6,))
        self.assertEqual(np.sum(student_version3), 44)

    def test_special_pattern(self):
        """ Test extracting all elements as presented in task 7. """

        student_version1 = numpy_basics.slicing_of_numpy_arrays_1d_07(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_1d_07(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_1d_07(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (4,))
        self.assertEqual(np.sum(student_version2), 22)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (5,))
        self.assertEqual(np.sum(student_version3), 2)


class TestArraySlicing2D(unittest.TestCase):
    """
    The class contains the remainder of test cases for task 1.6 - Array Slicing.
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the tests
        self.array1 = None
        self.array2 = np.arange(1, 17).reshape((4, 4))
        self.array3 = np.array([[-1, 15, 4, 6], [5, -4, 17, 0], [3, 3, -8, 6], [7, -1, -10, 2]])

    def test_first_element(self):
        """ Test extracting the first element of an array. """

        student_version1 = numpy_basics.slicing_of_numpy_arrays_2d_01(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_2d_01(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_2d_01(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertEqual(student_version2.shape, ())
        self.assertEqual(np.sum(student_version2), 1)

        # Test 3
        self.assertEqual(student_version3.shape, ())
        self.assertEqual(np.sum(student_version3), -1)

    def test_last_element(self):
        """ Test extracting the last element of an array. """

        student_version1 = numpy_basics.slicing_of_numpy_arrays_2d_02(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_2d_02(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_2d_02(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertEqual(student_version2.shape, ())
        self.assertEqual(np.sum(student_version2), 16)

        # Test 3
        self.assertEqual(student_version3.shape, ())
        self.assertEqual(np.sum(student_version3), 2)

    def test_first_half_row_wise(self):
        """ Test extracting the row-wise first half of elements of an array. """

        array2 = np.arange(1, 26).reshape((5, 5))
        array3 = np.array([[-1, 15, 4, 6, 1], [5, -4, 17, 0, 1], [3, 3, -8, 6, 1], [7, -1, -10, 2, 1], [1, 1, 1, 1, 1]])

        student_version1 = numpy_basics.slicing_of_numpy_arrays_2d_03(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_2d_03(array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_2d_03(array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (2, 5))
        self.assertEqual(np.sum(student_version2), 55)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (2, 5))
        self.assertEqual(np.sum(student_version3), 44)

    def test_second_half_row_wise(self):
        """ Test extracting the row-wise second half of elements of an array. """

        array2 = np.arange(1, 26).reshape((5, 5))
        array3 = np.array([[-1, 15, 4, 6, 1], [5, -4, 17, 0, 1], [3, 3, -8, 6, 1], [7, -1, -10, 2, 1], [1, 1, 1, 1, 1]])

        student_version1 = numpy_basics.slicing_of_numpy_arrays_2d_04(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_2d_04(array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_2d_04(array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (3, 5))
        self.assertEqual(np.sum(student_version2), 270)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (3, 5))
        self.assertEqual(np.sum(student_version3), 9)

    def test_first_half_column_wise(self):
        """ Test extracting the column-wise first half of elements of an array. """

        array2 = np.arange(1, 26).reshape((5, 5))
        array3 = np.array([[-1, 15, 4, 6, 1], [5, -4, 17, 0, 1], [3, 3, -8, 6, 1], [7, -1, -10, 2, 1], [1, 1, 1, 1, 1]])

        student_version1 = numpy_basics.slicing_of_numpy_arrays_2d_05(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_2d_05(array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_2d_05(array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (5, 2))
        self.assertEqual(np.sum(student_version2), 115)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (5, 2))
        self.assertEqual(np.sum(student_version3), 29)

    def test_second_half_column_wise(self):
        """ Test extracting the column-wise second half of elements of an array. """

        array2 = np.arange(1, 26).reshape((5, 5))
        array3 = np.array([[-1, 15, 4, 6, 1], [5, -4, 17, 0, 1], [3, 3, -8, 6, 1], [7, -1, -10, 2, 1], [1, 1, 1, 1, 1]])

        student_version1 = numpy_basics.slicing_of_numpy_arrays_2d_06(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_2d_06(array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_2d_06(array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (5, 3))
        self.assertEqual(np.sum(student_version2), 210)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (5, 3))
        self.assertEqual(np.sum(student_version3), 24)

    def test_odd_columns(self):
        """ Test extracting every element at an odd column in row-wise order of an array. """

        student_version1 = numpy_basics.slicing_of_numpy_arrays_2d_07(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_2d_07(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_2d_07(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (4, 2))
        self.assertEqual(np.sum(student_version2), 64)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (4, 2))
        self.assertEqual(np.sum(student_version3), 17)

    def test_even_columns(self):
        """ Test extracting every element at an even column in row-wise order of an array. """

        student_version1 = numpy_basics.slicing_of_numpy_arrays_2d_08(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_2d_08(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_2d_08(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (4, 2))
        self.assertEqual(np.sum(student_version2), 72)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (4, 2))
        self.assertEqual(np.sum(student_version3), 27)

    def test_diagonal(self):
        """ Test extracting every element on the diagonal of an array. """

        student_version1 = numpy_basics.slicing_of_numpy_arrays_2d_09(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_2d_09(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_2d_09(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (4, ))
        self.assertEqual(np.sum(student_version2), 34)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (4, ))
        self.assertEqual(np.sum(student_version3), -11)

    def test_diagonal_reverse(self):
        """ Test extracting every element on the reverse diagonal of an array. """

        student_version1 = numpy_basics.slicing_of_numpy_arrays_2d_10(self.array1)
        student_version2 = numpy_basics.slicing_of_numpy_arrays_2d_10(self.array2)
        student_version3 = numpy_basics.slicing_of_numpy_arrays_2d_10(self.array3)

        # Test 1
        self.assertEqual(student_version1, self.array1)

        # Test 2
        self.assertIsInstance(student_version2, np.ndarray)
        self.assertEqual(student_version2.shape, (4, ))
        self.assertEqual(np.sum(student_version2), 34)

        # Test 3
        self.assertIsInstance(student_version3, np.ndarray)
        self.assertEqual(student_version3.shape, (4, ))
        self.assertEqual(np.sum(student_version3), 33)


if __name__ == '__main__':
    # Instantiate the command line parser
    parser = argparse.ArgumentParser()

    # Add the option to run only a specific test case
    parser.add_argument('--test_case', help='Name of the test case you want to run')

    # Read the command line parameters
    args = parser.parse_args()

    # Run only a single test class
    if args.test_case:
        test_class = eval(args.test_case)
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        unittest.TextTestRunner().run(suite)

    # Run all test classes
    else:
        unittest.main(argv=[''], verbosity=1, exit=False)
