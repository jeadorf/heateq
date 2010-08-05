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

def add_stationary_opt(optparser):
    optparser.add_option(
        "-s",
        "--stationary",
        action="store_true",
        dest="stationary",
        default=False,
        help="Solve stationary heat equation.")

def add_n_opt(optparser):
    optparser.add_option(
        "-n",
        dest="n",
        default=10,
        type="int",
        help="Number of interior grid points."
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

def add_ts_opt(optparser):
    optparser.add_option(
        "--ts",
        action="store",
        dest="ts",
        type="float",
        default=1,
        help="Temperature value of left boundary."
    )

def add_te_opt(optparser):
    optparser.add_option(
        "--te",
        action="store",
        dest="te",
        type="float",
        default=-1,
        help="Temperature value of right boundary."
    )

def add_tinit_opt(optparser):
    optparser.add_option(
        "--tinit",
        action="store",
        dest="tinit",
        type="float_list",
        default="0,0,0",
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
    add_maxtime_opt(p)
    add_timestep_opt(p)
    add_locstep_opt(p)
    add_diffusivity_opt(p)
    add_ts_opt(p)
    add_te_opt(p)
    add_tinit_opt(p)
    add_pdf_opt(p)

    opts, args = p.parse_args()

    if opts.stationary:
        main_stationary(opts)
    else:
        main_instationary(opts)

def main_stationary(opts):
    t = sol.solve_stationary_1d(opts.ts, opts.te, opts.n)
    if opts.pdf != None:
        plot.gen_pdf_1d(t, opts.pdf)
    else:
        plot.show_win_1d_stationary(t)

def main_instationary(opts):
        sim  = sol.sim_heateq_1d(
                    opts.ts, opts.te, opts.tinit,
                    opts.diffusivity, opts.locstep, opts.timestep)
        win = gtk.Window()
        win.set_default_size(800, 100)
        tmin = min(opts.tinit)
        tmax = max(opts.tinit)
        tmin = min(tmin, opts.te, opts.ts)
        tmax = max(tmax, opts.te, opts.ts)
        print tmin
        print tmax
        tplot = plot.TemperaturePlot(tmin, tmax)
        win.add(tplot)
        win.connect("destroy", gtk.main_quit)
        win.show_all()
        stop = False

        def update(t):
            tplot.t = t
            tplot.redraw(None)

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

if __name__ == "__main__":
    main()
    
