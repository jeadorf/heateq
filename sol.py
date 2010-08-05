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

def sim_heateq_1d(ts, te, t_init, thermal_diffusivity=1, xstep=30, timestep=0.1):
    n = len(t_init)
    t = copy.copy(t_init)
    tm = 0
    xstepsq = 1. * xstep * xstep
    while True:
        yield ([ts(tm)] + t + [te(tm)], tm)
        # Compute derivative at all interior grid points
        dt = [ 0 for i in xrange(0, n) ]
        dt[0] = thermal_diffusivity * (ts(tm) - 2*t[0] + t[1]) / xstepsq
        for i in xrange(1, n-1):
            dt[i] = thermal_diffusivity * (t[i-1] - 2*t[i] + t[i+1]) / xstepsq
        dt[n-1] = thermal_diffusivity * (t[n-2] - 2*t[n-1] + te(tm)) / xstepsq
        # Euler step
        for i in xrange(0, n):
            t[i] += timestep * dt[i]
        tm += timestep

