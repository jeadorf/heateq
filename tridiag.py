#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Diag:
    def __init__(self, val):
        self.val = val
    def __getitem__(self, i):
        return self.val

def solve(d1, d2, d3, b, n):
    """Solves a tridiagonal system of n linear equations.  The matrix must not
    contain any zeros in the main diagonal or in one of the diagonals to the
    left or to the right of the main diagonal.
     
        #  #  0  0      A tridiagonal system matrix with non-zero elements (#)
        #  #  #  0      on the main diagonal and its both neighbor diagonals.
        0  #  #  #
        0  0  #  #
    
    The objects d1, d2, d3 contain the diagonal entries where d2 is the main
    diagonal. Object b represents the vector Ax.  It is sufficient that these
    objects provide a __getitem__ method in order to allow indexed access.
    Stability and numerical error have not been tested.
    """
    # Using gaussian elimination without any pivoting, adapted to a tridiagonal
    # system of linear equations (Ax = b).
    # A copy of b, will later serve as a container for the resulting vector.
    x = [ b[i] for i in xrange(0, n)]
    # Store the diagonal right to the main diagonal during gaussian
    # elimination.
    v = [ d3[i] for i in xrange(0, n)]
    # Keep track of the most recent element in the main diagonal (array d2 must
    # not be modified). Make sure that all operations are performed with
    # floating type.
    d = d2[0] * 1.
    # Iterate over each row, alternating between scaling and subtraction
    # operations.
    for i in xrange(0, n):
        # Choose the entry in the main diagonal as pivot. The entry left to the
        # pivot was eliminated at the end of the last iteration (or does not
        # exist at all).
        # Scale row such that the pivot will be 1. 
        x[i] /= d 
        v[i] /= d
        if i < n - 1:
            # Subtract row i from row i+1
            x[i+1] = x[i+1] - d1[i+1]*x[i]
            d = d2[i+1] - d1[i+1]*v[i]
    # Backsubstitution.
    for i in xrange(n-2, -1, -1):
        x[i] = x[i] - v[i]*x[i+1]
    return x

def solvec(dd1, dd2, dd3, b):
    """Solves a tridiagonal system of linear equations where a diagonal has
    the same entries in all rows. Example:
    
       -2  1  0  0
        1 -2  1  0
        0  1 -2  1
        0  0  1 -2
    
    See 'solve'."""
    return solve(Diag(dd1), Diag(dd2), Diag(dd3), b)

def test_solve():
    n = 4
    d1 = [1.0 for i in xrange(0, n)]
    d2 = [-2.0 for i in xrange(0, n)]
    d3 = [1.0 for i in xrange(0, n)]
    b = [5, -10, -5, 5]
    xcalc = solve(d1, d2, d3, b, n)
    x = [3, 11, 9, 2]
    eps = 1.e-8
    for i in xrange(0, n):
        assert abs(x[i] - xcalc[i]) < eps

def test_solve_c():
    n = 4
    b = [5, -10, -5, 5]
    xcalc = solve(Diag(1), Diag(-2), Diag(1), b, n)
    x = [3, 11, 9, 2]
    eps = 1.e-8
    for i in xrange(0, n):
        assert abs(x[i] - xcalc[i]) < eps


if __name__ == "__main__":
    d1 = [1.0 for i in xrange(0, 4)]
    d2 = [-2.0 for i in xrange(0, 4)]
    d3 = [1.0 for i in xrange(0, 4)]
    b = [5, -10, -5, 5]
    print solve(d1, d2, d3, b)
     
    d1 = [1.0 for i in xrange(0, 4000)]
    d2 = [-2.0 for i in xrange(0, 4000)]
    d3 = [1.0 for i in xrange(0, 4000)]
    b = [50] + [0] * 3998 + [-50]
    for i in xrange(0, 100):
        solve(d1, d2, d3, b)
