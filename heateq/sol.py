#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import heateqlapl


class InitConds2d:
    
    def __init__(self, m, n, top=None, right=None, bottom=None, left=None, interior=None):
        self.m = m
        self.n = n
        if top == None:
            self.top = const(numpy.zeros((n,)))
        else:
            self.top = top
        if bottom == None:
            self.bottom = const(numpy.zeros((n,)))
        else:
            self.bottom = bottom 
        if right == None:
            self.right = const(numpy.zeros((n,)))
        else:
            self.right = right
        if left == None:
            self.left = const(numpy.zeros((n,)))
        else:
            self.left = left
        if interior == None:
            self.interior = numpy.zeros((m, n))
        else:
            self.interior = interior
    
    def shape(self):
        return self.m, self.n

class InitConds1d:

    def __init__(self, n, left=None, right=None, interior=None):
        self.n = n
        if right == None:
            self.right = const(0)
        else:
            self.right = right
        if left == None:
            self.left = const(0)
        else:
            self.left = left
        if interior == None:
            self.interior = numpy.zeros((n,))
        else:
            self.interior = interior
 

def const(val):
    return (lambda tm: val)

def solve_stationary_1d(initconds):
    """Solves the stationary heat equation in one dimension.  The Dirichlet
    boundary conditions are given by values ts and te, describing the constant
    temperature values at grid points 0 and n+1.  The number of interior grid
    points is n > 1. Returns an array with n+2 elements, including the values
    at the boundaries."""
    return numpy.linspace(initconds.left(0), initconds.right(0), initconds.n)

def solve_stationary_2d(initconds):
    m, n = initconds.m, initconds.n
    ttop = initconds.top(0)
    tright = initconds.right(0)
    tbottom = initconds.bottom(0)
    tleft = initconds.left(0)
    a = laplacian(m, n)
    b = laplacian_b(ttop, tbottom, tleft, tright, m, n)
    x = numpy.linalg.solve(a, b)
    return x.reshape(m, n)

def laplacian(m, n):
    a = numpy.empty((m*n, m*n))
    for i in xrange(0, m*n):
        for j in xrange(0, m*n):
            if i == j:
                a[i, j] = -4
            elif abs(i / n - j / n) + abs(i % n - j % n) == 1:
                a[i, j] = 1
            else:
                a[i, j] = 0
    return a

def laplacian_b(ttop, tbottom, tleft, tright, m, n):
    b = numpy.empty((m*n,))
    for i in xrange(0, m*n):
        r = i / n
        c = i % n
        v = 0
        if r == 0:
            v -= ttop[c]
        elif r == m - 1:
            v -= tbottom[c]
        if c == 0:
            v -= tleft[r]
        elif c == n - 1:
            v -= tright[r]
        b[i] = v
    return b

def simulate_1d(initconds, diffy=1, delx=30, delt=0.1):
    n = initconds.n
    assert n > 1
    t = initconds.interior.copy()
    tm = 0
    delx2 = 1. * delx * delx
    dt = numpy.empty((n,))
    left = initconds.left
    right = initconds.right
    while True:
        yield t, tm
        # compute derivative
        dt[0] = diffy * (left(tm) - 2*t[0] + t[1]) / delx2
        for i in xrange(1, n-1):
            dt[i] = diffy * (t[i-1] - 2*t[i] + t[i+1]) / delx2
        dt[n-1] = diffy * (t[n-2] - 2*t[n-1] + right(tm)) / delx2
        # Euler
        t = t + delt * dt
        tm += delt

def simulate_2d(initconds, diffy=1, delx=30,  delt=0.1):
    m, n = initconds.shape()
    tm = 0
    t = initconds.interior.copy()
    dt = numpy.empty((m,n))
    delx2 = 1. * delx * delx
    while True:
        yield t, tm
        # Calculate second derivative
        heateqlapl.apply(t, dt, initconds.top(tm), initconds.bottom(tm), initconds.left(tm), initconds.right(tm))
        # Euler
        t = t + (1. * delt * diffy / delx2) * dt
        tm += delt

