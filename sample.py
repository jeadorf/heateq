#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sol
import plot
import random
import math

def kochtopf():
    m = 20 
    n = 40
    ttop = [0 for j in xrange(0, n) ]
    tbottom = [1 for j in xrange(0, n) ]
    tleft = [0 for i in xrange(0, m) ]
    tright = [0 for i in xrange(0, m) ]
    t = sol.solve_stationary_2d(ttop, tbottom, tleft, tright, m, n)
    plot.gen_pdf_2d(t, "kochtopf.pdf")

def randomheat():
    m = 20 
    n = 40
    ttop = [random.uniform(0, 1) for j in xrange(0, n) ]
    tbottom = [random.uniform(0, 1) for j in xrange(0, n) ]
    tleft = [random.uniform(0, 1)  for i in xrange(0, m) ]
    tright = [random.uniform(0, 1)  for i in xrange(0, m) ]
    t = sol.solve_stationary_2d(ttop, tbottom, tleft, tright, m, n)
    plot.gen_pdf_2d(t, "randomheat.pdf")

def sinfun():
    m = 40 
    n = 40
    ttop = [math.sin(math.pi * j / n) for j in xrange(0, n) ]
    tbottom = [math.sin(math.pi * j / n) for j in xrange(0, n) ]
    tleft = [math.sin(math.pi * (i + n/3) / n) for i in xrange(0, m) ]
    tright = [math.sin(math.pi * (i - n/3) / n) for i in xrange(0, m) ]
    t = sol.solve_stationary_2d(ttop, tbottom, tleft, tright, m, n)
    plot.gen_pdf_2d(t, "sinfun.pdf")

def fireinthecorner():
    m = 15 
    n = 15
    ttop = ([1 for j in xrange(0, n) ] +
           [0 for j in xrange(0, n) ] +
           [1 for j in xrange(0, n) ])
    tbottom = ([1 for j in xrange(0, n) ] +
           [0 for j in xrange(0, n) ] +
           [1 for j in xrange(0, n) ])
    tleft = ([1 for i in xrange(0, m) ] +
           [0 for i in xrange(0, m)  ] +
           [1 for i in xrange(0, m) ])
    tright = ([1 for i in xrange(0, m) ] +
           [0 for i in xrange(0, m) ] +
           [1 for i in xrange(0, m) ])
    t = sol.solve_stationary_2d(ttop, tbottom, tleft, tright, 3*m, 3*n)
    plot.gen_pdf_2d(t, "fireinthecorner.pdf")


def main():
    #kochtopf()
    #randomheat()
    sinfun()
    #fireinthecorner()

if __name__ == "__main__":
    main()
