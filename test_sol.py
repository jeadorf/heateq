#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sol import *
import time

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
    maxtm = 500
    for tt, tm in simulate_1d((lambda tm: ts), (lambda tm: te), [0, 0.2, 0.3], 300):
        t = tt
        if tm > maxtm:
            break
    assert abs(t[1]) < eps

def test_generate_matrix_2d():
    a = numpy.array(
        [[-4,  1,  0,  1,  0,  0],
         [ 1, -4,  1,  0,  1,  0],
         [ 0,  1, -4,  0,  0,  1],
         [ 1,  0,  0, -4,  1,  0],
         [ 0,  1,  0,  1, -4,  1],
         [ 0,  0,  1,  0,  1, -4]])
    c = numpy.array([[ generate_matrix_2d(i, j, 2, 3) for j in xrange(0, 6)] for i in xrange(0, 6) ])
    assert (a == c).all()

def test_generate_b_2d():
    tb_top = [1, 2, 3]
    tb_bottom = [7, 9, 10]
    tb_left = [4, 6]
    tb_right = [5, 11]
    b = numpy.array([generate_b_2d(i, tb_top, tb_bottom, tb_left, tb_right, 2, 3) for i in xrange(0, 6)])
    exp = numpy.array([-5, -2, -8, -13, -9, -21])
    assert (b == exp).all()

def test_speed_2d():
    st = time.clock()
    m, n = 50, 50
    ttop = numpy.zeros((n,))
    tbottom = numpy.ones((n,))
    tleft = numpy.zeros((m,))
    tright = numpy.zeros((m,))
    tinit = numpy.zeros((m,n))
    i = 0
    for t in simulate_2d(ttop, tbottom, tleft, tright, tinit):
        i += 1
        if i >= 20:
            break
    ct = time.clock()
    print "time=%.3fs,steps=%d,fps=%.2f" % (ct - st, i, i / (ct -st))

    
