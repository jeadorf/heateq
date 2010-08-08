#!/usr/bin/env python
# -*- coding: utf-8 -*-

from solver import *
from initconds import *
import time
import heateqlapl

eps = 1.e-8

def test_init_conds_2d():
    a = np.array([1, 2, 3])
    b = np.array([1, 2])
    top = lambda tm: 2 * a
    right = lambda tm: 3 * a
    bottom = lambda tm: 4 * b
    left = lambda tm: 5 * b
    ic = InitConds2d(2, 3, top, right, bottom, left)
    assert (ic.top(1) == 2 * a).all()
    assert (ic.right(1) == 3 * a).all()
    assert (ic.bottom(1) == 4 * b).all()
    assert (ic.left(1) == 5 * b).all()

def test_solve1d():
    ic = InitConds1d(5, const(5), const(-5))
    t = solve(ic)
    assert t[0] == 5 
    assert abs(t[2]) < eps
    assert t[-1] == -5

def test_if_simulate1d_converges():
    ts, te = 5, -5
    t = None
    maxtm = 500
    ic = InitConds1d(3, const(5), const(-5), np.array([0, 0.2, 0.3]))
    for tt, tm in simulate(ic, 300):
        t = tt
        if tm > maxtm:
            break
    assert abs(t[1]) < eps

def test_laplacian():
    a = np.array(
        [[-4,  1,  0,  1,  0,  0],
         [ 1, -4,  1,  0,  1,  0],
         [ 0,  1, -4,  0,  0,  1],
         [ 1,  0,  0, -4,  1,  0],
         [ 0,  1,  0,  1, -4,  1],
         [ 0,  0,  1,  0,  1, -4]])
    c = laplacian(2, 3)
    assert (a == c).all()

def test_laplacian_b():
    tb_top = [1, 2, 3]
    tb_bottom = [7, 9, 10]
    tb_left = [4, 6]
    tb_right = [5, 11]
    b = laplacian_b(tb_top, tb_bottom, tb_left, tb_right, 2, 3)
    exp = np.array([-5, -2, -8, -13, -9, -21])
    assert (b == exp).all()

def test_speed_2d():
    st = time.clock()
    m, n = 50, 50
    bottom = const(np.ones((n,)))
    ic = InitConds2d(m, n, bottom=bottom)
    i = 0
    for t in simulate(ic):
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
    a = laplacian(m, n)
    t = np.array([
        [4, 1, 0, 0],
        [3, 1, 0, 5],
        [0, -2, 1, 0]] , dtype=np.double)
    ttop = np.zeros((n,), dtype=np.double)
    tbottom = np.zeros((n,), dtype=np.double)
    tleft = np.zeros((m,), dtype=np.double)
    tright = np.zeros((m,), dtype=np.double)
    print "---------  a  ------------"
    print a 
    txx = np.zeros((m, n), dtype=np.double)
    heateqlapl.apply(t, txx, ttop, tbottom, tleft, tright)
    print "--------- txx ------------"
    print txx
    exp = np.dot(a, t.reshape((m*n, 1))).reshape((m, n))
    print "--------- exp ------------"
    print exp
    diff = exp - txx
    assert diff.max()**2 + diff.min()**2 < 1e-16;
