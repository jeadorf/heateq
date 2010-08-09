#!/usr/bin/env python
# -*- coding: utf-8 -*-

import solver
from initconds import InitConds2d, const
import render
import random
import math
import numpy

def kochtopf():
    print "generating 'kochtopf'"
    m = 20 
    n = 40
    ic = InitConds2d(m, n, bottom = const(numpy.ones((n,))))
    t = solver.solve(ic)
    context = render.RenderingContext(0, 1)
    render.render_pdf(t, context, "kochtopf.pdf")

def randomheat():
    print "generating 'randomheat'"
    m = 20 
    n = 40
    top = const(numpy.array([random.uniform(0, 1) for j in xrange(0, n) ]))
    right = const(numpy.array([random.uniform(0, 1)  for i in xrange(0, m) ]))
    bottom = const(numpy.array([random.uniform(0, 1) for j in xrange(0, n) ]))
    left = const(numpy.array([random.uniform(0, 1)  for i in xrange(0, m) ]))
    ic = InitConds2d(m, n, top, right, bottom, left)
    t = solver.solve(ic)
    context = render.RenderingContext(0, 1)
    render.render_pdf(t, context, "randomheat.pdf")

def sinfun():
    print "generating 'sinfun'"
    m = 50 
    n = 50
    top = const(numpy.array([math.sin(math.pi * j / n) for j in xrange(0, n) ]))
    bottom = const(numpy.array([math.sin(math.pi * j / n) for j in xrange(0, n) ]))
    left = const(numpy.array([math.sin(math.pi * (i + n/3) / n) for i in xrange(0, m) ]))
    right = const(numpy.array([math.sin(math.pi * (i - n/3) / n) for i in xrange(0, m) ]))
    ic = InitConds2d(m, n, top, right, bottom, left)
    t = solver.solve(ic)
    context = render.RenderingContext(-1, 1)
    render.render_pdf(t, context, "sinfun.pdf")

def fireinthecorner():
    print "generating 'fireintecorner'"
    m = 10
    n = 10
    top = const(numpy.array(
           [1 for j in xrange(0, n) ] +
           [0 for j in xrange(0, n) ] +
           [1 for j in xrange(0, n) ]))
    bottom = const(numpy.array(
           [1 for j in xrange(0, n) ] +
           [0 for j in xrange(0, n) ] +
           [1 for j in xrange(0, n) ]))
    left = const(numpy.array(
           [1 for i in xrange(0, m) ] +
           [0 for i in xrange(0, m) ] +
           [1 for i in xrange(0, m) ]))
    right = const(numpy.array(
           [1 for i in xrange(0, m) ] +
           [0 for i in xrange(0, m) ] +
           [1 for i in xrange(0, m) ]))
    ic = InitConds2d(3*m, 3*n, top, right, bottom, left)
    t = solver.solve(ic)
    context = render.RenderingContext(0, 1)
    render.render_pdf(t, context, "fireinthecorner.pdf")


def main():
    kochtopf()
    randomheat()
    sinfun()
    fireinthecorner()

if __name__ == "__main__":
    main()
