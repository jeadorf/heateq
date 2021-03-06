#!/usr/bin/env python
# -*- coding: utf-8 -*-

# internal package imports
from heateq.solver import simulate
from heateq.render import render2d
from heateq.initconds import InitConds2d, const_ones

# external package imports
import cairo
import numpy as np
import time

def test_render2d_speed():
    m, n = 30, 30
    it = InitConds2d(m, n, bottom=const_ones(n))
    max = 100
    imsf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 400, 400)
    ctx = cairo.Context(imsf)
    # with buffering
    stb = time.clock()
    i = 0
    c_buf = np.zeros((m, n))
    for t, tm in simulate(it):
        i += 1
        render2d(t, ctx, 0, 0, 400, 400, 0, 1, c_buf)
        if i >= max:
            break
    ctb = time.clock()
    # without buffering
    stn = time.clock()
    i = 0
    for t, tm in simulate(it):
        i += 1
        render2d(t, ctx, 0, 0, 400, 400, 0, 1)
        if i >= max:
            break
    ctn = time.clock()
    
    print "plotting statistics"
    print "time: %.3fs (without buffering: %.3fs)" % (ctb - stb, ctn - stn)
    if ctb - stb > 0 and ctn - stn > 0:
        print "frames per second: %.1f (without buffering: %.1f)" % (max / (ctb - stb), max / (ctn - stn))
    else:
        print "frames per second: n/a"
