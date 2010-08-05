#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tridiag
import copy

def solve_stationary_1d(ts, te, n):
    """Solves the stationary heat equation in one dimension.  The Dirichlet
    boundary conditions are given by values ts and te, describing the constant
    temperature values at grid points 0 and n+1.  The number of interior grid
    points is n > 1. Returns an array with n+2 elements, including the values
    at the boundaries."""
    return [ 1. * ts + 1. * i * (te - ts) / (n + 1) for i in xrange(0, n+2)]

def solve_stationary_1d_tridiag(ts, te, n):
    """Same as solve_stationary_1d, just using a tridiagonal system solver."""
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

def sim_heateq_1d(ts, te, t_init, thermal_diffusivity=1, xstep=30, timestep=0.1):
    n = len(t_init)
    t = copy.copy(t_init)
    xstepsq = 1. * xstep * xstep
    while True:
        yield [ts] + t + [te] 
        # Compute derivative at all interior grid points
        dt = [ 0 for i in xrange(0, n) ]
        dt[0] = thermal_diffusivity * (ts - 2*t[0] + t[1]) / xstepsq
        for i in xrange(1, n-1):
            dt[i] = thermal_diffusivity * (t[i-1] - 2*t[i] + t[i+1]) / xstepsq
        dt[n-1] = thermal_diffusivity * (t[n-2] - 2*t[n-1] + te) / xstepsq
        # Euler step
        for i in xrange(0, n):
            t[i] += timestep * dt[i]

def gen():
    while True:
        yield 1

if __name__ == "__main__":
    n = 10
    sim = sim_heateq_1d(-5, 0, [0 for i in xrange(0, n)], 10)
    i = 0
    for t in sim:
        print t
        i += 1
        if i > 5000:
            break
    print t
