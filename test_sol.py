#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sol import *

eps = 1.e-8

def test_solve_stationary_1d():
    ts, te, n = 5, -5, 3 
    t = solve_stationary_1d(ts, te, n)
    assert t[0] == ts
    assert abs(t[2]) < eps
    assert t[n+1] == te

def test_if_simulate_1d_converges():
    ts, te = 5, -5
    t = None
    maxtm = 5000
    for tt, tm in simulate_1d((lambda tm: ts), (lambda tm: te), [0, 0.2, 0.3], 300):
        t = tt
        if tm > maxtm:
            break
    assert t[0] == ts
    assert abs(t[2]) < eps
    assert t[4] == te

def test_generate_matrix_2d():
    a = numpy.matrix(
        [[-4,  1,  0,  1,  0,  0],
         [ 1, -4,  1,  0,  1,  0],
         [ 0,  1, -4,  0,  0,  1],
         [ 1,  0,  0, -4,  1,  0],
         [ 0,  1,  0,  1, -4,  1],
         [ 0,  0,  1,  0,  1, -4]])
    c = numpy.matrix([[ generate_matrix_2d(i, j, 2, 3) for j in xrange(0, 6)] for i in xrange(0, 6) ])
    assert (a == c).all()

def test_generate_b_2d():
    tb_top = [1, 2, 3]
    tb_bottom = [7, 9, 10]
    tb_left = [4, 6]
    tb_right = [5, 11]
    b = numpy.matrix([generate_b_2d(i, tb_top, tb_bottom, tb_left, tb_right, 2, 3) for i in xrange(0, 6)])
    exp = numpy.matrix([-5, -2, -8, -13, -9, -21])
    assert (b == exp).all()
    
