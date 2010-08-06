#!/usr/bin/env python
# -*- coding: utf-8 -*-

import plot
import plot2d
import time
import sol
import numpy
import cairo

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
    stc = time.clock()
    for t, tm in sol.simulate_2d(ttop, tbottom, tleft, tright, tinit):
        i += 1
        plot.plot_2d(t, ctx, 0, 0, 400, 400, 0, 1, False)
        if i >= max:
            break
    ctc = time.clock()
    i = 0
    stp = time.clock()
    for t, tm in sol.simulate_2d(ttop, tbottom, tleft, tright, tinit):
        i += 1
        plot2d.plot2d(t, ctx, 0, 0, 400, 400, 0, 1)
        if i >= max:
            break
    ctp = time.clock()
    print "Python time=%.3fs" % (ctc - stc)
    print "C Ext. time=%.3fs" % (ctp - stp)
 
