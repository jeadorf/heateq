#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sol 
import plot
import optparse
import copy

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
    optparser = optparse.OptionParser(option_class=ExtOption)
    add_stationary_opt(optparser)
    add_n_opt(optparser)
    add_timestep_opt(optparser)
    add_locstep_opt(optparser)
    add_ts_opt(optparser)
    add_te_opt(optparser)
    add_tinit_opt(optparser)
    add_diffusivity_opt(optparser)
    add_pdf_opt(optparser)

    opts, args = optparser.parse_args()

    if opts.stationary:
        t = sol.solve_stationary_1d(opts.ts, opts.te, opts.n)
        if opts.pdf != None:
            plot.gen_pdf_1d(t, opts.pdf)
        else:
            plot.show_win_1d_stationary(t)
    else:
        sim = sol.sim_heateq_1d(opts.ts, opts.te, opts.tinit, opts.diffusivity, opts.locstep, opts.timestep)
        i = 0
        for t, time in sim:
            print t
            print time
            i += 1
            if i > 20:
                break

if __name__ == "__main__":
    main()
    
