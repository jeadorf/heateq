#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import heateqlapl
import scipy.sparse.linalg.isolve

def solve(ic):
    """Solve the stationary heat equation to the given boundary values. Assumes
    that the temperature at the boundaries is constant and simply grabs the
    temperature values at time 0.

    """
    if ic.dim == 1:
        return _solve1d(ic)
    elif ic.dim == 2:
        return _solve2d(ic)
    else:
        raise ValueError("Unsupported dimension: %d" % ic.dim)

def _solve1d(ic):
    return np.linspace(ic.left(0), ic.right(0), ic.n)

def _solve2d(ic):
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
    """Generate a matrix that approximates the effect of the 2-dimensional Laplace operator using a five-point stencil.
    Such a matrix with m=3, n=3 looks like this:

     -4   1   0   1   0   0   0   0   0 
      1  -4   1   0   1   0   0   0   0 
      0   1  -4   0   0   1   0   0   0 
      1   0   0  -4   1   0   1   0   0 
      0   1   0   1  -4   1   0   1   0 
      0   0   1   0   1  -4   0   0   1 
      0   0   0   1   0   0  -4   1   0 
      0   0   0   0   1   0   1  -4   1 
      0   0   0   0   0   1   0   1  -4

    m -- number of rows
    n -- number of columns
    
    """
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
    """Create the vector b on the right-hand side of a linear system of
    equations Ax = b, where A is the matrix that approximates the 2-dimensional
    Laplace operator with a five-point stencil, and x is the vector of
    temperature values.  This vector solely depends on the number of grid
    points and the boundary values.

    m, n -- number of rows and columns
    ttop, tright, tbottom, tleft  -- temperature values at the boundaries

    """
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

def simulate(ic, dfy=1, dx=30, dt=0.1):
    """Simulate heat diffusion in a homogeneous 2-dimensional area.
    
    ic  -- a description of initial values and boundary conditions
    dfy -- thermal diffusivity
    dx  -- grid spacing, same for both directions
    dt  -- time-step, the period of time between two simulation states 

    Choosing the right combination of values for dfy, dx, and dt is crucial
    as to avoid oscillations.  For the magnitude of the change applied in each
    simulation step is controlled by the factor (dfy*dt)/(dx*dx) the following
    choices might lead to oscillations:

    - large thermal diffusivity
    - big time-steps
    - small grid spacing
    
    This has severe implications.  Doubling the resolution of space requires
    about quartering the time-step.  Note that from a computational view,
    values for thermal diffusivity and time-step are interchangeable.  Yet,
    diffusivity is a bound variable while timestep and grid spacing are subject
    to available computing resources.

    """
    if ic.dim == 1:
        return _simulate1d(ic, dfy, dx, dt)
    elif ic.dim == 2:
        return _simulate2d(ic, dfy, dx, dt)
    else:
        raise ValueError("Unsupported dimension: %d" % ic.dim)

def _simulate1d(ic, dfy, dx, dt):
    n = ic.n
    assert n > 1
    t = ic.interior.copy()
    tm = 0
    da = (1. * dfy * dt) / (dx * dx)
    t_t = np.empty((n,))
    left = ic.left
    right = ic.right
    while True:
        yield t, tm
        # Calculate t_t from current temperature values 
        t_t[0] = left(tm) - 2*t[0] + t[1]
        for i in xrange(1, n-1):
            t_t[i] = t[i-1] - 2*t[i] + t[i+1]
        t_t[n-1] = t[n-2] - 2*t[n-1] + right(tm)
        # Euler
        t = t + da * t_t
        tm += dt

def _simulate2d(ic, dfy, dx,  dt):
    m, n = ic.shape()
    tm = 0
    t = ic.interior.copy()
    t_t = np.empty((m,n))
    da = (1. * dfy * dt) / (dx * dx)
    while True:
        yield t, tm
        # Calculate t_t from current temperature values 
        heateqlapl.apply(t, t_t, ic.top(tm), ic.bottom(tm), ic.left(tm), ic.right(tm))
        # Euler
        t = t + da * t_t
        tm += dt
