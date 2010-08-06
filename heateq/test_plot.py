#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import sol
import numpy
import cairo
from plot import *

def test_plot2d_speed():
    m, n = 30, 30
    ttop = numpy.zeros((n,))
    tbottom = numpy.ones((n,))
    tleft = numpy.zeros((m,))
    tright = numpy.zeros((m,))
    tinit = numpy.zeros((m,n))
    max = 100
    imsf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 400, 400)
    ctx = cairo.Context(imsf)
    i = 0
    st = time.clock()
    for t, tm in sol.simulate_2d((lambda tm: ttop), (lambda tm: tbottom), (lambda tm: tleft), (lambda tm: tright), tinit):
        i += 1
        plot_2d(t, ctx, 0, 0, 400, 400, 0, 1)
        if i >= max:
            break
    ct = time.clock()
    print "plotting statistics"
    print "time: %.3fs" % (ct - st)
    if ct - st != 0:
        print "frames per second: %.1f" % (i / (ct - st) )
    else:
        print "frames per second: n/a"


 
def foo():
    print "bar"
