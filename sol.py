#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import numpy, numpy.linalg

def solve_stationary_1d(ts, te, n):
    """Solves the stationary heat equation in one dimension.  The Dirichlet
    boundary conditions are given by values ts and te, describing the constant
    temperature values at grid points 0 and n+1.  The number of interior grid
    points is n > 1. Returns an array with n+2 elements, including the values
    at the boundaries."""
    return [ 1. * ts + 1. * i * (te - ts) / (n + 1) for i in xrange(0, n+2)]

def solve_stationary_2d(tb_top, tb_bottom, tb_left, tb_right, m, n):
    # Matrix generator
    a = numpy.matrix([ [ generate_matrix_2d(i, j, m, n) for j in xrange(0, m*n) ] for i in xrange(0, m*n) ])
    b = numpy.matrix([ [ generate_b_2d(i, tb_top, tb_bottom, tb_left, tb_right, m, n) ] for i in xrange(0, m*n)])
    x = numpy.linalg.solve(a, b)
    t = [ [ x.item(i * n + j) for j in xrange(0, n) ] for i in xrange(0, m) ]
    return t

def generate_matrix_2d(i, j, m, n):
    if i == j:
        return -4
    elif abs(i / n - j / n) + abs(i % n - j % n) == 1:
        return 1
    else:
        return 0

def generate_b_2d(i, tb_top, tb_bottom, tb_left, tb_right, m, n):
    r = i / n
    c = i % n
    b = 0
    if r == 0:
        b -= tb_top[c]
    elif r == m - 1:
        b -= tb_bottom[c]
    if c == 0:
        b -= tb_left[r]
    elif c == n - 1:
        b -= tb_right[r]
    return b

def simulate_1d(ts, te, t_init, diffusivity=1, delx=30, delt=0.1):
    n = len(t_init)
    assert n > 1
    t = copy.copy(t_init)
    tm = 0
    delx2 = 1. * delx * delx 
    dt = [ 0 for i in xrange(0, n) ]
    while True:
        yield ([ts(tm)] + t + [te(tm)], tm)
        # Compute derivative at all interior grid points
        dt[0] = diffusivity * (ts(tm) - 2*t[0] + t[1]) / delx2
        for i in xrange(1, n-1):
            dt[i] = diffusivity * (t[i-1] - 2*t[i] + t[i+1]) / delx2 
        dt[n-1] = diffusivity * (t[n-2] - 2*t[n-1] + te(tm)) / delx2
        # Euler step
        for i in xrange(0, n):
            t[i] += delt * dt[i]
        tm += delt

def simulate_2d(ttop, tbottom, tleft, tright, tinit, diffusivity=1, delx=30,  delt=0.1):
    m = len(tinit)
    n = len(tinit[0])
    t = copy.deepcopy(tinit)
    tm = 0
    delx2 = 1. * delx * delx
    dt = [ [ 0 for j in xrange(0, n) ] for i in xrange(0, m) ]
    while True:
        yield t, tm
        # Compute derivative at all interior grid points
        dt[0][0] = diffusivity * (ttop[0] + t[1][0] - 4*t[0][0] + tleft[0] + t[0][1]) / delx2
        for j in xrange(1, n-1):
            dt[0][j] = diffusivity * (ttop[j] + t[1][j] - 4*t[0][j] + t[0][j-1] + t[0][j+1]) / delx2
        dt[0][n-1] = diffusivity * (ttop[n-1] + t[1][n-1] - 4*t[0][n-1] + tright[0] + t[0][n-2]) / delx2
        for i in xrange(1, m-1):
            dt[i][0] = diffusivity * (t[i-1][j] + t[i+1][j] - 4*t[i][j] + t[i][j-1] + tright[i]) / delx2
            for j in xrange(1, n-1):
                dt[i][j] = diffusivity * (t[i-1][j] + t[i+1][j] - 4*t[i][j] + t[i][j-1] + t[i][j+1]) / delx2
            dt[i][n-1] = diffusivity * (t[i-1][n-1] + t[i+1][n-1] - 4*t[i][n-1] + t[i][n-2] + tright[i]) / delx2
        dt[m-1][0] = diffusivity * (tbottom[0] + t[m-2][0] - 4*t[m-1][0] + tleft[m-1] + t[m-1][1]) / delx2
        for j in xrange(1, n-1):
            dt[m-1][j] = diffusivity * (tbottom[j] + t[m-1][j] - 4*t[m-1][j] + t[m-1][j-1] + t[m-1][j+1]) / delx2
        dt[m-1][n-1] = diffusivity * (tbottom[n-1] + t[m-2][n-1] - 4*t[m-1][n-1] + tright[m-1] + t[m-1][n-2]) / delx2
        # Euler step
        for i in xrange(0, m):
            for j in xrange(0, n):
                t[i][j] += delt * dt[i][j]
        tm += delt

