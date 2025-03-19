#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
This script was written as a test module for the 'numpy_applied.ipynb'-jupyter notebook.
That notebook tests your knowledge about the basic numpy functions by letting you apply them to real world tasks.
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
import numpy_applied


class TestCheckerBoard(unittest.TestCase):
    """
    The class contains all test cases for task "1 - Checkerboard Pattern".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Load the reference images
        self.checker_expected1 = np.load("../data/numpy_applied/reference_checker1.npz")['data']
        self.checker_expected2 = np.load("../data/numpy_applied/reference_checker2.npz")['data']
        self.checker_expected3 = np.load("../data/numpy_applied/reference_checker3.npz")['data']

        # Initialize the used sizes and tile sizes for the reference images
        self.size1, self.tile_size1 = 10, 1
        self.size2, self.tile_size2 = 100, 25
        self.size3, self.tile_size3 = 200, 25

    def test_attributes(self):
        """
        Test whether the students implemented an attribute 'pattern'.
        """

        # Create the student version of the pattern
        student_checker = numpy_applied.CheckerBoard(self.size1, self.tile_size1)

        try:
            student_version = student_checker.pattern
            pass

        except AttributeError:
            self.assertTrue(False, "Class CheckerBoard does not contain an attribute 'pattern'!")

    def test_relation_size_to_tilesize(self):
        """
        Test whether the students pattern omit non-matching size and tile size parameter.
        """

        # create the student version of the pattern with non-matching size and tile size parameter
        student_checker = numpy_applied.CheckerBoard(10, 3)

        try:
            student_checker.draw()
            student_version = student_checker.pattern

            # Check whether the pattern does contain values
            self.assertIsNone(student_version, "Pattern was created despite having a non-suitable size and tile size "
                                               "relation!")

        except:
            # Extract the pattern
            student_version = student_checker.pattern

            # Check whether the pattern does contain values
            self.assertIsNone(student_version, "Pattern was created despite having a non-suitable size and tile size "
                                               "relation!")

    def test_pattern1(self):
        """
        Test whether the students pattern matches the expected one.
        """

        # Create the student version of the pattern
        student_checker = numpy_applied.CheckerBoard(self.size1, self.tile_size1)
        student_checker.draw()
        student_version = student_checker.pattern

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.checker_expected1.shape,
                         "Shape of pattern is not correct!")
        self.assertEqual(student_version[0, 0], 0, "Pattern is not correct!")
        self.assertEqual(np.min(student_version), 0, "Pattern is not correct!")
        self.assertEqual(np.max(student_version), 1, "Pattern is not correct!")
        self.assertEqual(float(np.sum(student_version)), float(np.sum(self.checker_expected1)),
                         "Pattern is not correct!")

    def test_pattern2(self):
        """
        Test whether the students pattern matches the expected one with different parameters.
        """

        # create the student version of the pattern
        student_checker = numpy_applied.CheckerBoard(self.size2, self.tile_size2)
        student_checker.draw()
        student_version = student_checker.pattern

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.checker_expected2.shape,
                         "Shape of pattern is not correct!")
        self.assertEqual(student_version[0, 0], 0, "Pattern is not correct!")
        self.assertEqual(np.min(student_version), 0, "Pattern is not correct!")
        self.assertEqual(np.max(student_version), 1, "Pattern is not correct!")
        self.assertEqual(float(np.sum(student_version)), float(np.sum(self.checker_expected2)),
                         "Pattern is not correct!")

    def test_pattern3(self):
        """
        Test whether the students pattern matches the expected one with different parameters.
        """

        # create the student version of the pattern
        student_checker = numpy_applied.CheckerBoard(self.size3, self.tile_size3)
        student_checker.draw()
        student_version = student_checker.pattern

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.checker_expected3.shape,
                         "Shape of pattern is not correct!")
        self.assertEqual(student_version[0, 0], 0, "Pattern is not correct!")
        self.assertEqual(np.min(student_version), 0, "Pattern is not correct!")
        self.assertEqual(np.max(student_version), 1, "Pattern is not correct!")
        self.assertEqual(float(np.sum(student_version)), float(np.sum(self.checker_expected3)),
                         "Pattern is not correct!")


class TestRectangle(unittest.TestCase):
    """
    The class contains all test cases for task "2.1 - Create the Rectangle Shape".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Load the reference images
        self.rectangle_expected1 = np.load("../data/numpy_applied/reference_rectangle1.npz")['data']
        self.rectangle_expected2 = np.load("../data/numpy_applied/reference_rectangle2.npz")['data']
        self.rectangle_expected3 = np.load("../data/numpy_applied/reference_rectangle3.npz")['data']

        # Initialize the used sizes and coordinates for the reference images
        self.size1, self.x1y1_1, self.x2y2_1 = 10, (0, 0), (10, 10)
        self.size2, self.x1y1_2, self.x2y2_2 = 100, (75, 40), (25, 60)
        self.size3, self.x1y1_3, self.x2y2_3 = 200, (20, 180), (180, 20)

    def _IoU(self, a1, a2):
        # Utility function returning the intersection over union value
        intersection = np.sum(a1 * a2)
        union = a1 + a2
        union = np.sum(union.astype(np.bool_))
        iou = intersection/union

        return iou

    def test_attributes(self):
        """
        Test whether the students implemented an attribute 'pattern'.
        """

        # Create the student version of the pattern
        student_rectangle = numpy_applied.Rectangle(self.size1, self.x1y1_1, self.x2y2_1)

        try:
            student_version = student_rectangle.pattern
            pass

        except AttributeError:
            self.assertTrue(False, "Class CheckerBoard does not contain an attribute 'pattern'!")

    def test_pattern1(self):
        """
        Test whether the students pattern matches the expected one.
        """

        # Create the student version of the pattern
        student_rectangle = numpy_applied.Rectangle(self.size1, self.x1y1_1, self.x2y2_1)
        student_rectangle.draw()
        student_version = student_rectangle.pattern

        # Calculate the IOU
        iou = self._IoU(student_version, self.rectangle_expected1)

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.rectangle_expected1.shape,
                         "Shape of pattern is not correct!")
        self.assertEqual(student_version[0, 0], 1, "Pattern is not correct!")
        self.assertEqual(np.min(student_version), 1, "Pattern is not correct!")
        self.assertEqual(np.max(student_version), 1, "Pattern is not correct!")
        self.assertEqual(float(np.sum(student_version)), float(np.sum(self.rectangle_expected1)),
                         "Pattern is not correct!")
        self.assertAlmostEqual(iou, 1.0, 1, "Pattern is not correct!")

    def test_pattern2(self):
        """
        Test whether the students pattern matches the expected one with different parameters.
        """

        # Create the student version of the pattern
        student_rectangle = numpy_applied.Rectangle(self.size2, self.x1y1_2, self.x2y2_2)
        student_rectangle.draw()
        student_version = student_rectangle.pattern

        # Calculate the IOU
        iou = self._IoU(student_version, self.rectangle_expected2)

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.rectangle_expected2.shape,
                         "Shape of pattern is not correct!")
        self.assertEqual(student_version[0, 0], 0, "Pattern is not correct!")
        self.assertEqual(np.min(student_version), 0, "Pattern is not correct!")
        self.assertEqual(np.max(student_version), 1, "Pattern is not correct!")
        self.assertEqual(float(np.sum(student_version)), float(np.sum(self.rectangle_expected2)),
                         "Pattern is not correct!")
        self.assertAlmostEqual(iou, 1.0, 1, "Pattern is not correct!")

    def test_pattern3(self):
        """
        Test whether the students pattern matches the expected one with different parameters.
        """

        # Create the student version of the pattern
        student_rectangle = numpy_applied.Rectangle(self.size3, self.x1y1_3, self.x2y2_3)
        student_rectangle.draw()
        student_version = student_rectangle.pattern

        # Calculate the IOU
        iou = self._IoU(student_version, self.rectangle_expected3)

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.rectangle_expected3.shape,
                         "Shape of pattern is not correct!")
        self.assertEqual(student_version[0, 0], 0, "Pattern is not correct!")
        self.assertEqual(np.min(student_version), 0, "Pattern is not correct!")
        self.assertEqual(np.max(student_version), 1, "Pattern is not correct!")
        self.assertEqual(float(np.sum(student_version)), float(np.sum(self.rectangle_expected3)),
                         "Pattern is not correct!")
        self.assertAlmostEqual(iou, 1.0, 1, "Pattern is not correct!")


class TestCircle(unittest.TestCase):
    """
    The class contains all test cases for task "2.2 - Create the Circle Shape".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Load the reference images
        self.circle_expected1 = np.load("../data/numpy_applied/reference_circle1.npz")['data']
        self.circle_expected2 = np.load("../data/numpy_applied/reference_circle2.npz")['data']
        self.circle_expected3 = np.load("../data/numpy_applied/reference_circle3.npz")['data']

        # Initialize the used sizes, radia and center for the reference images
        self.size1, self.radius1, self.center1 = 10, 0, (5, 2)
        self.size2, self.radius2, self.center2 = 100, 20, (0, 0)
        self.size3, self.radius3, self.center3 = 200, 50, (70, 120)

    def _IoU(self, a1, a2):
        # Utility function returning the intersection over union value
        intersection = np.sum(a1 * a2)
        union = a1 + a2
        union = np.sum(union.astype(np.bool_))
        iou = intersection/union

        return iou

    def test_attributes(self):
        """
        Test whether the students implemented an attribute 'pattern'.
        """

        # Create the student version of the pattern
        student_circle = numpy_applied.Circle(self.size1, self.radius1, self.center1)

        try:
            student_version = student_circle.pattern
            pass

        except AttributeError:
            self.assertTrue(False, "Class CheckerBoard does not contain an attribute 'pattern'!")

    def test_pattern1(self):
        """
        Test whether the students pattern matches the expected one.
        """

        # Create the student version of the pattern
        student_circle = numpy_applied.Circle(self.size1, self.radius1, self.center1)
        student_circle.draw()
        student_version = student_circle.pattern

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.circle_expected1.shape,
                         "Shape of pattern is not correct!")
        self.assertEqual(student_version[0, 0], 0, "Pattern is not correct!")
        self.assertEqual(np.min(student_version), 0, "Pattern is not correct!")
        self.assertEqual(np.max(student_version), 0, "Pattern is not correct!")
        self.assertEqual(float(np.sum(student_version)), float(np.sum(self.circle_expected1)),
                         "Pattern is not correct!")

    def test_pattern2(self):
        """
        Test whether the students pattern matches the expected one with different parameters.
        """

        # Create the student version of the pattern
        student_circle = numpy_applied.Circle(self.size2, self.radius2, self.center2)
        student_circle.draw()
        student_version = student_circle.pattern

        # Calculate the IOU
        iou = self._IoU(student_version, self.circle_expected2)

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.circle_expected2.shape,
                         "Shape of pattern is not correct!")
        self.assertEqual(student_version[0, 0], 1, "Pattern is not correct!")
        self.assertEqual(np.min(student_version), 0, "Pattern is not correct!")
        self.assertEqual(np.max(student_version), 1, "Pattern is not correct!")
        self.assertEqual(float(np.sum(student_version)), float(np.sum(self.circle_expected2)),
                         "Pattern is not correct!")
        self.assertAlmostEqual(iou, 1.0, 1, "Pattern is not correct!")

    def test_pattern3(self):
        """
        Test whether the students pattern matches the expected one with different parameters.
        """

        # Create the student version of the pattern
        student_circle = numpy_applied.Circle(self.size3, self.radius3, self.center3)
        student_circle.draw()
        student_version = student_circle.pattern

        # Calculate the IOU
        iou = self._IoU(student_version, self.circle_expected3)

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.circle_expected3.shape,
                         "Shape of pattern is not correct!")
        self.assertEqual(student_version[0, 0], 0, "Pattern is not correct!")
        self.assertEqual(np.min(student_version), 0, "Pattern is not correct!")
        self.assertEqual(np.max(student_version), 1, "Pattern is not correct!")
        self.assertEqual(float(np.sum(student_version)), float(np.sum(self.circle_expected3)),
                         "Pattern is not correct!")
        self.assertAlmostEqual(iou, 1.0, 1, "Pattern is not correct!")


class TestShapes(unittest.TestCase):
    """
    The class contains all test cases for task "2.3 - Combine and Plot the Shapes".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Load the reference images
        self.shapes_expected1 = np.load("../data/numpy_applied/reference_shapes1.npz")['data']
        self.shapes_expected2 = np.load("../data/numpy_applied/reference_shapes2.npz")['data']
        self.shapes_expected3 = np.load("../data/numpy_applied/reference_shapes3.npz")['data']

        # Initialize the used sizes, coordinates, radia and center for the reference images
        self.size1, self.x1y1_1, self.x2y2_1, self.radius1, self.center1 = 10, (0, 0), (10, 10), 0, (5, 2)
        self.size2, self.x1y1_2, self.x2y2_2, self.radius2, self.center2 = 100, (75, 40), (25, 60), 20, (0, 0)
        self.size3, self.x1y1_3, self.x2y2_3, self.radius3, self.center3 = 200, (20, 180), (180, 20), 50, (70, 120)

    def _IoU(self, a1, a2):
        # Utility function returning the intersection over union value
        intersection = np.sum(a1 * a2)
        union = a1 + a2
        union = np.sum(union.astype(np.bool_))
        iou = intersection/union

        return iou

    def test_pattern1(self):
        """
        Test whether the students pattern matches the expected one.
        """

        # Create the student version of the pattern
        student_version = numpy_applied.combine_and_plot_shapes(self.size1, self.x1y1_1, self.x2y2_1, self.radius1,
                                                                self.center1)

        # Calculate the IOU
        iou = self._IoU(student_version, self.shapes_expected1)

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.shapes_expected1.shape,
                         "Shape of pattern is not correct!")
        self.assertEqual(student_version[0, 0], 1, "Pattern is not correct!")
        self.assertEqual(np.min(student_version), 1, "Pattern is not correct!")
        self.assertEqual(np.max(student_version), 1, "Pattern is not correct!")
        self.assertEqual(float(np.sum(student_version)), float(np.sum(self.shapes_expected1)),
                         "Pattern is not correct!")
        self.assertAlmostEqual(iou, 1.0, 1, "Pattern is not correct!")

    def test_pattern2(self):
        """
        Test whether the students pattern matches the expected one with different parameters.
        """

        # Create the student version of the pattern
        student_version = numpy_applied.combine_and_plot_shapes(self.size2, self.x1y1_2, self.x2y2_2, self.radius2,
                                                                self.center2)

        # Calculate the IOU
        iou = self._IoU(student_version, self.shapes_expected2)

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.shapes_expected2.shape,
                         "Shape of pattern is not correct!")
        self.assertEqual(student_version[0, 0], 1, "Pattern is not correct!")
        self.assertEqual(np.min(student_version), 0, "Pattern is not correct!")
        self.assertEqual(np.max(student_version), 1, "Pattern is not correct!")
        self.assertEqual(float(np.sum(student_version)), float(np.sum(self.shapes_expected2)),
                         "Pattern is not correct!")
        self.assertAlmostEqual(iou, 1.0, 1, "Pattern is not correct!")

    def test_pattern3(self):
        """
        Test whether the students pattern matches the expected one with different parameters.
        """

        # Create the student version of the pattern
        student_version = numpy_applied.combine_and_plot_shapes(self.size3, self.x1y1_3, self.x2y2_3, self.radius3,
                                                                self.center3)

        # Calculate the IOU
        iou = self._IoU(student_version, self.shapes_expected3)

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.shapes_expected3.shape,
                         "Shape of pattern is not correct!")
        self.assertEqual(student_version[0, 0], 0, "Pattern is not correct!")
        self.assertEqual(np.min(student_version), 0, "Pattern is not correct!")
        self.assertEqual(np.max(student_version), 1, "Pattern is not correct!")
        self.assertEqual(float(np.sum(student_version)), float(np.sum(self.shapes_expected3)),
                         "Pattern is not correct!")
        self.assertAlmostEqual(iou, 1.0, 1, "Pattern is not correct!")


class TestSpectrum(unittest.TestCase):
    """
    The class contains all test cases for task "3 - Color Spectrum (Optional)".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Load the reference images
        self.spectrum_expected1 = np.load("../data/numpy_applied/reference_spectrum1.npz")['data']
        self.spectrum_expected2 = np.load("../data/numpy_applied/reference_spectrum2.npz")['data']
        self.spectrum_expected3 = np.load("../data/numpy_applied/reference_spectrum3.npz")['data']

        # Initialize the used sizes
        self.size1 = 10
        self.size2 = 100
        self.size3 = 255

    def test_attributes(self):
        """
        Test whether the students implemented an attribute 'pattern'.
        """

        # Create the student version of the pattern
        student_spectrum = numpy_applied.Spectrum(self.size1)

        try:
            student_version = student_spectrum.pattern
            pass

        except AttributeError:
            self.assertTrue(False, "Class CheckerBoard does not contain an attribute 'pattern'!")

    def test_pattern1(self):
        """
        Test whether the students pattern matches the expected one.
        """

        # Create the student version of the pattern
        student_spectrum = numpy_applied.Spectrum(self.size1)
        student_spectrum.draw()
        student_version = student_spectrum.pattern

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.spectrum_expected1.shape,
                         "Shape of pattern is not correct!")
        np.testing.assert_almost_equal(student_version, self.spectrum_expected1, decimal=2)

    def test_pattern2(self):
        """
        Test whether the students pattern matches the expected one with different parameters.
        """

        # Create the student version of the pattern
        student_spectrum = numpy_applied.Spectrum(self.size2)
        student_spectrum.draw()
        student_version = student_spectrum.pattern

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.spectrum_expected2.shape,
                         "Shape of pattern is not correct!")
        np.testing.assert_almost_equal(student_version, self.spectrum_expected2, decimal=2)

    def test_pattern3(self):
        """
        Test whether the students pattern matches the expected one with different parameters.
        """

        # Create the student version of the pattern
        student_spectrum = numpy_applied.Spectrum(self.size3)
        student_spectrum.draw()
        student_version = student_spectrum.pattern

        # Compare the student version to the expected output
        self.assertEqual(student_version.shape, self.spectrum_expected3.shape,
                         "Shape of pattern is not correct!")
        np.testing.assert_almost_equal(student_version, self.spectrum_expected3, decimal=2)


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
#%%
