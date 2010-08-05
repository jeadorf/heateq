#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tridiag

def solve_stationary_1d(ts, te, n):
    """Solves the stationary heat equation in one dimension.  The Dirichlet
    boundary conditions are given by values ts and te, describing the constant
    temperature values at grid points 0 and n+1.  The number of interior grid
    points is n > 1. Returns an array with n+2 elements, including the values
    at the boundaries."""
    assert n > 1
    d1 = tridiag.Diag(1)
    d2 = tridiag.Diag(-2)
    d3 = tridiag.Diag(1)
    class BVector:
        def __getitem__(self, i):
            if i == 0:
                return -ts
            elif i == n - 1:
                return -te
            else:
                return 0
    b = BVector()
    t = tridiag.solve(d1, d2, d3, b, n)
    return [ts] + t + [te]

