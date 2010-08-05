#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sol import *

def test_solve_stationary_1d():
    ts, te, n = 5, -5, 3 
    t = solve_stationary_1d(ts, te, n)
    eps = 1.e-8
    print t
    assert t[0] == ts
    assert abs(t[2]) < eps
    assert t[n+1] == te

