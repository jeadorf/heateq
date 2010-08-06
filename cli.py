#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sol 
import plot
import optparse
import copy
import threading
import gobject
import gtk
import time
import sys

def add_stationary_opt(optparser):
    optparser.add_option(
        "-s",
        "--stationary",
        action="store_true",
        dest="stationary",
        default=False,
        help="Solve stationary heat equation.")

def add_dimensions_opt(optparser):
    optparser.add_option(
        "-D",
        "--dimensions",
        dest="dimensions",
        default=1,
        type="int",
        help="Number of dimensions (1 or 2)."
    )

def add_n_opt(optparser):
    optparser.add_option(
        "-n",
        dest="n",
        default=10,
        type="int",
        help="Number of interior grid points on the x-axis."
    )

def add_m_opt(optparser):
    optparser.add_option(
        "-m",
        dest="m",
        default=10,
        type="int",
        help="Number of interior grid points on the y-axis."
    )

def add_maxtime_opt(optparser):
    optparser.add_option(
        "--maxtime",
        action="store",
        dest="maxtime",
        type="float",
        default=-1,
        help="Maximum simulation time."
    )

def add_timestep_opt(optparser):
    optparser.add_option(
        "--timestep",
        action="store",
        dest="timestep",
        type="float",
        default=0.001,
        help="Timespan between two simulation steps."
    )

def add_locstep_opt(optparser):
    optparser.add_option(
        "--locstep",
        action="store",
        dest="locstep",
        type="float",
        default=10,
        help="Distance between two grid points."
    )

def add_diffusivity_opt(optparser):
    optparser.add_option(
        "-d",
        "--diffusivity",
        action="store",
        dest="diffusivity",
        type="float",
        default=1,
        help="Thermal diffusivity of the simulated area."
    )

def add_tleft_opt(optparser):
    optparser.add_option(
        "--tleft",
        action="store",
        dest="tleft",
        type="float_list",
        default=[1],
        help="Temperature values at the left boundary."
    )

def add_tright_opt(optparser):
    optparser.add_option(
        "--tright",
        action="store",
        dest="tright",
        type="float_list",
        default=[-1],
        help="Temperature values at the right boundary."
    )

def add_ttop_opt(optparser):
    optparser.add_option(
        "--ttop",
        action="store",
        dest="ttop",
        type="float_list",
        default=[-1],
        help="Temperature values at the upper boundary."
    )

def add_tbottom_opt(optparser):
    optparser.add_option(
        "--tbottom",
        action="store",
        dest="tbottom",
        type="float_list",
        default=[1],
        help="Temperature values at the lower boundary."
    )

def add_tinit_opt(optparser):
    optparser.add_option(
        "--tinit",
        action="store",
        dest="tinit",
        type="float_list",
        default=[0, 0, 0],
        help="Initial temperature at interior grid points."
    )

def add_pdf_opt(optparser):
    optparser.add_option(
        "--pdf",
        action="store",
        dest="pdf",
        help="Save result to PDF. Only admittable for stationary problems."
    )


def check_float_list(option, opt, value):
    try:
        lst = [] 
        for v in value.split(","):
            lst.append(float(v))
        return lst 
    except ValueError:
        raise optparse.OptionValueError(
            "option %s: invalid float list value: %r" % (opt, value))

class ExtOption(optparse.Option):
    TYPES = optparse.Option.TYPES + ("float_list",)
    TYPE_CHECKER = copy.copy(optparse.Option.TYPE_CHECKER)
    TYPE_CHECKER["float_list"] = check_float_list

def main():
    p = optparse.OptionParser(option_class=ExtOption)
    add_stationary_opt(p)
    add_n_opt(p)
    add_m_opt(p)
    add_dimensions_opt(p)
    add_maxtime_opt(p)
    add_timestep_opt(p)
    add_locstep_opt(p)
    add_diffusivity_opt(p)
    add_ttop_opt(p)
    add_tbottom_opt(p)
    add_tleft_opt(p)
    add_tright_opt(p)
    add_tinit_opt(p)
    add_pdf_opt(p)

    opts, args = p.parse_args()

    if opts.dimensions == 2:
        if opts.stationary:
            main_stationary_2d(opts)
        else:
            main_instationary_2d(opts)
    else:
        if opts.stationary:
            main_stationary_1d(opts)
        else:
            main_instationary_1d(opts)

def main_stationary_1d(opts):
    t = sol.solve_stationary_1d(opts.tleft[0], opts.tright[0], opts.n)
    if opts.pdf != None:
        plot.gen_pdf_1d(t, opts.pdf)
    else:
        plot.show_win_1d_stationary(t)

def main_instationary_1d(opts):
        sim  = sol.simulate_1d(
                (lambda tm: opts.tleft[0]), (lambda tm: opts.tright[0]), opts.tinit,
                opts.diffusivity, opts.locstep, opts.timestep)
        win = gtk.Window()
        win.set_default_size(800, 100)
        tmin = min(opts.tinit)
        tmax = max(opts.tinit)
        tmin = min(tmin, opts.tleft[0], opts.tright[0])
        tmax = max(tmax, opts.tleft[0], opts.tright[0])
        tplot = plot.TemperaturePlot(tmin, tmax)
        win.add(tplot)
        win.connect("destroy", gtk.main_quit)
        win.show_all()
        stop = False

        def update(t):
            tplot.t = t
            tplot.queue_draw()

        def run_simulation(): 
            old_wtm = time.time()
            old_tm = 0
            for t, tm in sim:
                if time.time() - old_wtm > 1./30:
                    gobject.idle_add(update, copy.copy(t))
                    old_wtm = time.time()
                time.sleep(max(1/100., (tm - old_tm)))
                old_tm = tm
                if opts.maxtime >= 0 and tm > opts.maxtime or stop:
                    break

        gtk.gdk.threads_init()
        threading.Thread(target=run_simulation).start()
        gtk.main()
        stop = True

def main_stationary_2d(opts):
    m = opts.m 
    n = opts.n
    if len(opts.ttop) < n:
        opts.ttop = [0 for j in xrange(0, n)]
    if len(opts.tbottom) < n:
        opts.tbottom = [1 for j in xrange(0, n)]
    if len(opts.tleft) < m:
        opts.tleft = [0 for i in xrange(0, m)]
    if len(opts.tright) < m:
        opts.tright = [0 for i in xrange(0, m)]
    t = sol.solve_stationary_2d(opts.ttop, opts.tbottom, opts.tleft, opts.tright, m, n)
    if opts.pdf == None:
        plot.show_win_2d_stationary(t)
    else:
        plot.gen_pdf_2d(t, opts.pdf)
import math
def main_instationary_2d(opts):
        m = opts.m 
        n = opts.n
        opts.tinit = [ [ 0 for j in xrange(0, n) ] for i in xrange(0, m) ]
        opts.ttop = [ math.sin(0.5 * math.pi + 1.0 * j / n * math.pi) for j in xrange(0, n) ]
        opts.tbottom = [ math.sin(- 0.5 * math.pi + 1.0 * j / n * math.pi) for j in xrange(0, n) ]
        opts.tleft = [ math.sin(- 0.25 * math.pi + 1.0 * i / n * math.pi) for i in xrange(0, m) ]
        opts.tright = [ math.sin( 0.25 * math.pi + 1.0 * i / n * math.pi) for i in xrange(0, m) ]
        win = gtk.Window()
        win.set_default_size(400, 400)
        tmin = sys.maxint
        tmax = -sys.maxint
        for i in xrange(0, m):
            tmin = min(tmin, min(opts.tinit[i]))
            tmax = max(tmax, max(opts.tinit[i]))
        tmin = min(tmin, min(opts.ttop))
        tmin = min(tmin, min(opts.tbottom))
        tmin = min(tmin, min(opts.tleft))
        tmin = min(tmin, min(opts.tright))
        tmax = max(tmax, max(opts.ttop))
        tmax = max(tmax, max(opts.tbottom))
        tmax = max(tmax, max(opts.tleft))
        tmax = max(tmax, max(opts.tright))
        
        sim = sol.simulate_2d(opts.ttop, opts.tbottom, opts.tleft, opts.tright, opts.tinit,
                              opts.diffusivity, opts.locstep, opts.timestep)
            
        tplot = plot.TemperaturePlot(tmin, tmax, dim=2)
        win.add(tplot)
        win.connect("destroy", gtk.main_quit)
        win.show_all()
        stop = False

        def update(t):
            tplot.t = t
            tplot.queue_draw()

        def run_simulation(): 
            old_wtm = time.time()
            old_tm = 0
            for t, tm in sim:
                if time.time() - old_wtm > .5:
                    gobject.idle_add(update, [ copy.copy(t[i]) for i in xrange(0, len(t)) ])
                    old_wtm = time.time()
                time.sleep(max(1/100., (tm - old_tm)))
                old_tm = tm
                if opts.maxtime >= 0 and tm > opts.maxtime or stop:
                    break

        gtk.gdk.threads_init()
        threading.Thread(target=run_simulation).start()
        gtk.main()
        stop = True


if __name__ == "__main__":
    main()
    
