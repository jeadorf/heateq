#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sol import *
import time
import laplace2d

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
    for t in simulate_2d((lambda tm: ttop), (lambda tm: tbottom), (lambda tm: tleft), (lambda tm: tright), tinit):
        i += 1
        if i >= 20000:
            break
    ct = time.clock()
    print "simulator statistics"
    print "time: %.3fs" % (ct - st)
    print "frames: %d" % i
    if ct - st != 0:
        print "frames per second: %.2f [thousand]" % (i / (ct - st) / 1e3)
        print "pixel updates per second: %.2f [million]" % (i*m*n / (ct - st) / 1e6)
    else:
        print "frames per second: n/a"
        print "pixel updates per second: n/a"

def test_laplace2d():
    m = 3
    n = 4 
    a = numpy.empty((m*n, m*n))
    for i in xrange(0, m*n):
        for j in xrange(0, m*n):
            a[i, j] = generate_matrix_2d(i, j, m, n)
    t = numpy.array([
        [4, 1, 0, 0],
        [3, 1, 0, 5],
        [0, -2, 1, 0]] , dtype=numpy.double)
    ttop = numpy.zeros((n,), dtype=numpy.double)
    tbottom = numpy.zeros((n,), dtype=numpy.double)
    tleft = numpy.zeros((m,), dtype=numpy.double)
    tright = numpy.zeros((m,), dtype=numpy.double)
    print "---------  a  ------------"
    print a 
    txx = numpy.zeros((m, n), dtype=numpy.double)
    laplace2d.apply(t, txx, ttop, tbottom, tleft, tright)
    print "--------- txx ------------"
    print txx
    exp = numpy.dot(a, t.reshape((m*n, 1))).reshape((m, n))
    print "--------- exp ------------"
    print exp
    diff = exp - txx
    assert diff.max()**2 + diff.min()**2 < 1e-16;
