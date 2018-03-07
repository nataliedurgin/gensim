#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
import logging
import unittest
import numpy as np
from scipy.special import psi  # gamma function utils

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import gensim.matutils as matutils



# we'll define known, good (slow) version of functions here
# and compare results from these functions vs. cython ones
def logsumexp(x):
    """Log of sum of exponentials.

    Parameters
    ----------
    x : numpy.ndarray
        Input 2d matrix.

    Returns
    -------
    float
        log of sum of exponentials of elements in `x`.

    Warnings
    --------
    By performance reasons, doesn't support NaNs or 1d, 3d, etc arrays like :func:`scipy.special.logsumexp`.

    """
    x_max = np.max(x)
    x = np.log(np.sum(np.exp(x - x_max)))
    x += x_max

    return x


def mean_absolute_difference(a, b):
    """Mean absolute difference between two arrays.

    Parameters
    ----------
    a : numpy.ndarray
        Input 1d array.
    b : numpy.ndarray
        Input 1d array.

    Returns
    -------
    float
        mean(abs(a - b)).

    """
    return np.mean(np.abs(a - b))


def dirichlet_expectation(alpha):
    """For a vector :math:`\\theta \sim Dir(\\alpha)`, compute :math:`E[log \\theta]`.

    Parameters
    ----------
    alpha : numpy.ndarray
        Dirichlet parameter 2d matrix or 1d vector, if 2d - each row is treated as a separate parameter vector.

    Returns
    -------
    numpy.ndarray:
        :math:`E[log \\theta]`

    """
    if len(alpha.shape) == 1:
        result = psi(alpha) - psi(np.sum(alpha))
    else:
        result = psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]
    return result.astype(alpha.dtype, copy=False)  # keep the same precision as input


dirichlet_expectation_1d = dirichlet_expectation
dirichlet_expectation_2d = dirichlet_expectation


class TestLdaModelInner(unittest.TestCase):
    def setUp(self):
        self.random_state = np.random.RandomState()
        self.num_runs = 100  # test functions with *num_runs* random inputs
        self.num_topics = 100

    def testLogSumExp(self):
        # test logsumexp
        rs = self.random_state

        for dtype in [np.float16, np.float32, np.float64]:
            for i in range(self.num_runs):
                input = rs.uniform(-1000, 1000, size=(self.num_topics, 1))

                known_good = logsumexp(input)
                test_values = matutils.logsumexp(input)

                msg = "logsumexp failed for dtype={}".format(dtype)
                self.assertTrue(np.allclose(known_good, test_values), msg)

    def testMeanAbsoluteDifference(self):
        # test mean_absolute_difference
        rs = self.random_state

        for dtype in [np.float16, np.float32, np.float64]:
            for i in range(self.num_runs):
                input1 = rs.uniform(-10000, 10000, size=(self.num_topics,))
                input2 = rs.uniform(-10000, 10000, size=(self.num_topics,))

                known_good = mean_absolute_difference(input1, input2)
                test_values = matutils.mean_absolute_difference(input1, input2)

                msg = "mean_absolute_difference failed for dtype={}".format(dtype)
                self.assertTrue(np.allclose(known_good, test_values), msg)

    def testDirichletExpectation(self):
        # test dirichlet_expectation
        rs = self.random_state

        for dtype in [np.float16, np.float32, np.float64]:
            for i in range(self.num_runs):
                # 1 dimensional case
                input_1d = rs.uniform(.01, 10000, size=(self.num_topics,))
                known_good = dirichlet_expectation(input_1d)
                test_values = matutils.dirichlet_expectation(input_1d)

                msg = "dirichlet_expectation_1d failed for dtype={}".format(dtype)
                self.assertTrue(np.allclose(known_good, test_values), msg)

                # 2 dimensional case
                input_2d = rs.uniform(.01, 10000, size=(1, self.num_topics,))
                known_good = dirichlet_expectation(input_2d)
                test_values = matutils.dirichlet_expectation(input_2d)

                msg = "dirichlet_expectation_2d failed for dtype={}".format(dtype)
                self.assertTrue(np.allclose(known_good, test_values), msg)


class TestLevenshteinSimilarityMatrix(unittest.TestCase):
    """Test levenshtein_similarity_matrix returns expected results."""

    """    
    For an explanation of the Levenshtein distance algorithm see, for example: 
    https://people.cs.pitt.edu/~kirk/cs1501/Pruhs/Spring2006/assignments/editdistance/Levenshtein%20Distance.htm
    Relevant for manually computing edit-distance in test cases
    """

    def setUp(self):
        from gensim.test.utils import common_corpus, common_dictionary
        # Example to highlight that the tfidf reordering happens successfully
        self.mini_texts = [['ab'], ['abc', 'ab'], ['bcd', 'ab']]
        self.mini_dict = Dictionary(self.mini_texts)
        self.mini_corpus = [self.mini_dict.doc2bow(text)
                            for text in self.mini_texts]
        self.mini_tfidf = TfidfModel(self.mini_corpus)

        self.corpus = common_corpus
        self.dictionary = common_dictionary
        self.tfidf = TfidfModel(common_corpus)

        # Some different initializations of the levenshtein similarity matrix
        self.similarity_matrix = matutils.levenshtein_similarity_matrix(
            self.dictionary).todense()
        self.similarity_matrix_alpha = matutils.levenshtein_similarity_matrix(
            self.mini_dict, alpha=1).todense()
        self.similarity_matrix_beta = matutils.levenshtein_similarity_matrix(
            self.mini_dict, beta=1).todense()
        self.similarity_matrix_mini = matutils.levenshtein_similarity_matrix(
            self.mini_dict).todense()
        self.similarity_matrix_tfidf = matutils.levenshtein_similarity_matrix(
            self.mini_dict, tfidf=self.mini_tfidf).todense()

        # Some explicitly computed edit-distance matrices
        self.mini_lev_raw = np.array([[0., 1 - 1 / 3., 1 - 3 / 3.],
                                      [1 - 1 / 3., 0., 1 - 2 / 3.],
                                      [1 - 3 / 3., 1 - 2 / 3., 0.]])
        self.tfidf_lev_raw = np.array([[0., 1 - 2 / 3., 1 - 1 / 3.],
                                       [1 - 2 / 3., 0., 1 - 3 / 3.],
                                       [1 - 1 / 3., 1 - 3 / 3., 0.]])

    def test_formula(self):
        # Check that the formula is working correctly,
        # Manually compute the term similarity matrix for the mini_dict
        # Assume the default term ordering
        mini_lev = 1.8 * self.mini_lev_raw ** 5 + np.identity(3)

        # for numerical tolerances of the np.allclose() method see
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.allclose.html
        self.assertTrue(np.allclose(mini_lev, self.similarity_matrix_mini))

    def test_matrix_symmetry(self):
        # checking symmetry
        self.assertTrue(
            (self.similarity_matrix.T == self.similarity_matrix).all())

    def test_ones_along_diagonal(self):
        # Check existence of ones on the diagonal
        self.assertTrue(
            (np.diag(self.similarity_matrix) ==
             np.ones(self.similarity_matrix.shape[0])).all())

    def test_alpha(self):
        # checking that alpha works as expected
        mini_lev = 1 * self.mini_lev_raw ** 5 + np.identity(3)
        self.assertTrue(np.allclose(mini_lev, self.similarity_matrix_alpha))

    def test_beta(self):
        # checking that beta works as expected
        mini_lev = 1.8 * self.mini_lev_raw ** 1 + np.identity(3)
        self.assertTrue(np.allclose(mini_lev, self.similarity_matrix_beta))

    def test_tfidf_term_ordering(self):
        # Check to make sure supplying tfidf reordered the scores
        # This test is a bit fragile but it is something
        tfidf_lev = 1.8 * self.tfidf_lev_raw ** 5 + np.identity(3)
        self.assertTrue(np.allclose(tfidf_lev, self.similarity_matrix_tfidf))

    def test_at_most_one(self):
        # Checking that all matrix entries are at most one when alpha=1
        self.assertTrue((self.similarity_matrix_alpha <= 1).all())



if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
