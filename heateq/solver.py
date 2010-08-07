#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import heateqlapl

def solve1d(ic):
    return np.linspace(ic.left(0), ic.right(0), ic.n)

def solve2d(ic):
    m, n = ic.m, ic.n
    ttop = ic.top(0)
    tright = ic.right(0)
    tbottom = ic.bottom(0)
    tleft = ic.left(0)
    a = laplacian(m, n)
    b = laplacian_b(ttop, tbottom, tleft, tright, m, n)
    x = np.linalg.solve(a, b)
    return x.reshape(m, n)

def laplacian(m, n):
    a = np.empty((m*n, m*n))
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
    b = np.empty((m*n,))
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

def simulate1d(ic, diffy=1, delx=30, delt=0.1):
    n = ic.n
    assert n > 1
    t = ic.interior.copy()
    tm = 0
    delx2 = 1. * delx * delx
    dt = np.empty((n,))
    left = ic.left
    right = ic.right
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

def simulate2d(ic, diffy=1, delx=30,  delt=0.1):
    m, n = ic.shape()
    tm = 0
    t = ic.interior.copy()
    dt = np.empty((m,n))
    delx2 = 1. * delx * delx
    while True:
        yield t, tm
        # Calculate second derivative
        heateqlapl.apply(t, dt, ic.top(tm), ic.bottom(tm), ic.left(tm), ic.right(tm))
        # Euler
        t = t + (1. * delt * diffy / delx2) * dt
        tm += delt

