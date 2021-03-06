#!/usr/bin/env python
# -*- coding: utf-8 -*-

# internal package imports
from heateq.solver import simulate, solve
from heateq.initconds import InitConds2d, InitConds1d, const
from heateq.render import render1d, render2d, render_pdf, render_win, RenderingContext, TPlot1d, TPlot2d

# external package imports
import copy
import gobject
import gtk
import math
import numpy as np
import optparse
import sys
import threading
import time

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
        default=1000,
        help="Thermal diffusivity of the simulated area."
    )

def add_tleft_opt(optparser):
    optparser.add_option(
        "--tleft",
        action="store",
        dest="tleft",
        type="ndarray",
        default=[1],
        help="Temperature values at the left boundary."
    )

def add_tright_opt(optparser):
    optparser.add_option(
        "--tright",
        action="store",
        dest="tright",
        type="ndarray",
        default=[-1],
        help="Temperature values at the right boundary."
    )

def add_ttop_opt(optparser):
    optparser.add_option(
        "--ttop",
        action="store",
        dest="ttop",
        type="ndarray",
        default=[-1],
        help="Temperature values at the upper boundary."
    )

def add_tbottom_opt(optparser):
    optparser.add_option(
        "--tbottom",
        action="store",
        dest="tbottom",
        type="ndarray",
        default=[1],
        help="Temperature values at the lower boundary."
    )

def add_tinit_opt(optparser):
    optparser.add_option(
        "--tinit",
        action="store",
        dest="tinit",
        type="ndarray",
        default=np.zeros((1,)),
        help="Initial temperature at interior grid points."
    )

def add_pdf_opt(optparser):
    optparser.add_option(
        "--pdf",
        action="store",
        dest="pdf",
        help="Save result to PDF. Only admittable for stationary problems."
    )


def check_ndarray(option, opt, value):
    try:
        return np.fromstring(value, sep=",")
    except ValueError:
        raise optparse.OptionValueError(
            "option %s: invalid ndarray value: %r" % (opt, value))

class ExtOption(optparse.Option):
    TYPES = optparse.Option.TYPES + ("ndarray",)
    TYPE_CHECKER = copy.copy(optparse.Option.TYPE_CHECKER)
    TYPE_CHECKER["ndarray"] = check_ndarray

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
    ic = InitConds1d(opts.n, const(opts.tleft[0]), const(opts.tright[0]))
    t = solve(ic)
    if opts.pdf != None:
        context = RenderingContext(t.min(), t.max())
        render_pdf(t, context, opts.pdf)
    else:
        render_win(t)

def main_instationary_1d(opts):
        if len(opts.tinit) < opts.n:
            opts.tinit = np.zeros((opts.n, ))
        ic = InitConds1d(opts.n, const(opts.tleft[0]), const(opts.tright[0]), opts.tinit)
        sim  = simulate(ic, opts.diffusivity, opts.locstep, opts.timestep)
        win = gtk.Window()
        win.set_default_size(800, 100)
        tmin = min(opts.tinit)
        tmax = max(opts.tinit)
        tmin = min(tmin, opts.tleft[0], opts.tright[0])
        tmax = max(tmax, opts.tleft[0], opts.tright[0])
        context = RenderingContext(tmin, tmax)
        tplot = TPlot1d(context)
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
                    gobject.idle_add(update, t.copy())
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
        opts.ttop = np.zeros((n,))
    if len(opts.tbottom) < n:
        opts.tbottom = np.ones((n,))
    if len(opts.tleft) < m:
        opts.tleft = np.zeros((m,))
    if len(opts.tright) < m:
        opts.tright = np.zeros((m,))
    ic = InitConds2d(m, n, const(opts.ttop), const(opts.tright), const(opts.tbottom), const(opts.tleft))
    t = solve(ic)
    if opts.pdf == None:
        render_win(t)
    else:
        # todo: find better procedure to find tmin and tmax
        tmin = min(opts.tinit.min(), opts.ttop.min(), opts.tbottom.min(), opts.tleft.min(), opts.tright.min())
        tmax = max(opts.tinit.max(), opts.ttop.max(), opts.tbottom.max(), opts.tleft.max(), opts.tright.max())
        context = RenderingContext(tmin, tmax)
        render_pdf(t, context, opts.pdf)

def main_instationary_2d(opts):
        m = opts.m
        n = opts.n
        opts.tinit = np.array([ [ 0 for j in xrange(0, n) ] for i in xrange(0, m) ])
        opts.ttop = np.array([ math.sin(0.5 * math.pi + 1.0 * j / n * math.pi) for j in xrange(0, n) ])
        opts.tright = np.array([ math.sin( 0.25 * math.pi + 1.0 * i / n * math.pi) for i in xrange(0, m) ])
        opts.tbottom = np.array([ math.sin(- 0.5 * math.pi + 1.0 * j / n * math.pi) for j in xrange(0, n) ])
        opts.tleft = np.array([ math.sin(- 0.25 * math.pi + 1.0 * i / n * math.pi) for i in xrange(0, m) ])
        win = gtk.Window()
        win.set_default_size(400, 400)
        
        # todo: find better procedure to find tmin and tmax
        tmin = min(opts.tinit.min(), opts.ttop.min(), opts.tbottom.min(), opts.tleft.min(), opts.tright.min())
        tmax = max(opts.tinit.max(), opts.ttop.max(), opts.tbottom.max(), opts.tleft.max(), opts.tright.max())
        w = lambda tt: (lambda tm: 2 * math.sin(tm / 50) * tt)
        ic = InitConds2d(m, n, w(opts.ttop), w(opts.tright), w(opts.tbottom), w(opts.tleft), opts.tinit)

        sim = simulate(ic, opts.diffusivity, opts.locstep, opts.timestep)

        context = RenderingContext(tmin, tmax)
        tplot = TPlot2d(context)
        win.add(tplot)
        win.connect("destroy", gtk.main_quit)
        win.show_all()
        stop = False

        def update(t):
            tplot.t = t
            tplot.window.invalidate_rect(tplot.get_allocation(), True)

        def run_simulation():
            old_wtm = time.time()
            old_tm = 0
            i = 0
            for t, tm in sim:
                if time.time() - old_wtm > 0.15:
                    gobject.idle_add(update, t.copy())
                    old_wtm = time.time()
                if i > 150:
                    time.sleep(50)
                    i = 0
                old_tm = tm
                if opts.maxtime >= 0 and tm > opts.maxtime or stop:
                    break

        gtk.gdk.threads_init()
        threading.Thread(target=run_simulation).start()
        gtk.main()
        stop = True

if __name__ == "__main__":
    main()

