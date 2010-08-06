#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

def solve_stationary_1d(ts, te, n):
    """Solves the stationary heat equation in one dimension.  The Dirichlet
    boundary conditions are given by values ts and te, describing the constant
    temperature values at grid points 0 and n+1.  The number of interior grid
    points is n > 1. Returns an array with n+2 elements, including the values
    at the boundaries."""
    return numpy.linspace(ts, te, n+2)

def solve_stationary_2d(ttop, tbottom, tleft, tright, m, n):
    a = numpy.empty((m*n, m*n))
    b = numpy.empty((m*n,))
    for i in xrange(0, m*n):
        for j in xrange(0, m*n):
            a[i, j] = generate_matrix_2d(i, j, m, n)
        b[i] = generate_b_2d(i, ttop, tbottom, tleft, tright, m, n)
    x = numpy.linalg.solve(a, b)
    return x.reshape(m, n)

def generate_matrix_2d(i, j, m, n):
    if i == j:
        return -4
    elif abs(i / n - j / n) + abs(i % n - j % n) == 1:
        return 1
    else:
        return 0

def generate_b_2d(i, ttop, tbottom, tleft, tright, m, n):
    r = i / n
    c = i % n
    b = 0
    if r == 0:
        b -= ttop[c]
    elif r == m - 1:
        b -= tbottom[c]
    if c == 0:
        b -= tleft[r]
    elif c == n - 1:
        b -= tright[r]
    return b

def simulate_1d(ts, te, tinit, diffy=1, delx=30, delt=0.1):
    n = len(tinit)
    assert n > 1
    t = numpy.array(tinit)
    tm = 0
    delx2 = 1. * delx * delx
    dt = numpy.empty((n,))
    while True:
        yield t, tm
        # compute derivative
        dt[0] = diffy * (ts(tm) - 2*t[0] + t[1]) / delx2
        for i in xrange(1, n-1):
            dt[i] = diffy * (t[i-1] - 2*t[i] + t[i+1]) / delx2
        dt[n-1] = diffy * (t[n-2] - 2*t[n-1] + te(tm)) / delx2
        # Euler
        t = t + delt * dt
        tm += delt

def simulate_2d(ttop, tbottom, tleft, tright, tinit, diffy=1, delx=30,  delt=0.1):
    m = len(tinit)
    n = len(tinit[0])
    tm = 0
    t = numpy.zeros((m, n))
    for i in xrange(0, m):
        for j in xrange(0, n):
            t[i, j] = tinit[i][j]
    dt = numpy.empty((m,n))
    delx2 = 1. * delx * delx
    while True:
        yield t, tm
        # compute derivative
        dt[0, 0] = diffy * (ttop[0] + t[1, 0] + tleft[0] + t[0, 1] - 4 * t[0, 0]) / delx2
        for j in xrange(1, n-1):
            dt[0, j] = diffy * (ttop[j] + t[1, j] + t[0, j-1] + t[0, j+1] - 4*t[0, j]) / delx2
        dt[0, n-1] = diffy * (ttop[n-1] + t[1, n-1] + t[0, n-2] + tright[0] - 4*t[0, n-1]) / delx2
        for i in xrange(1, m-1):
            dt[i, 0] = diffy * (t[i-1, 0] + t[i+1, 0] + tleft[i] + t[i, 1] - 4*t[i, 0]) / delx2
            for j in xrange(1, n-1):
               dt[i, j] = diffy * (t[i-1, j] + t[i+1, j] + t[i, j-1] + t[i, j+1] - 4*t[i, j]) / delx2
            dt[i, n-1] = diffy * (t[i-1, n-1] + t[i+1, n-1] + t[i, n-2] + tright[i] - 4*t[i, n-1]) / delx2
        dt[m-1, 0] = diffy * (t[m-1, 0] + tbottom[0] + tleft[0] + t[m-1, 1] - 4*t[m-1, 0]) / delx2
        for j in xrange(1, n-1):
            dt[m-1, j] = diffy * (t[m-1, j] + tbottom[j] + t[m-1, j-1] + t[m-1, j+1]- 4*t[m-1, j]) / delx2
        dt[m-1, n-1] = diffy * (t[m-1, n-1] + tbottom[n-1] + t[m-1, n-2] + tright[m-1]- 4*t[m-1, n-1]) / delx2
        # Euler
        t = t + delt * dt
        tm += delt

