#!/usr/bin/env python
# -*- coding: utf-8 -*-

import optparse

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
        help="Number of interior grid points."
    )

def add_timestep_opt(optparser):
    optparser.add_option(
        "--timestep",
        action="store",
        dest="timestep",
        default=0.001,
        help="Timespan between two simulation steps."
    )

def add_locstep_opt(optparser):
    optparser.add_option(
        "--locstep",
        action="store",
        dest="locstep",
        default=10,
        help="Distance between two grid points."
    )

def add_ts_opt(optparser):
    optparser.add_option(
        "--ts",
        action="store",
        dest="ts",
        default=1,
        help="Temperature value of left boundary."
    )

def add_te_opt(optparser):
    optparser.add_option(
        "--te",
        action="store",
        dest="te",
        default=-1,
        help="Temperature value of right boundary."
    )

def add_tinit_opt(optparser):
    optparser.add_option(
        "--tinit",
        action="store",
        dest="tinit",
        default=[0, 0, 0],
        help="Initial temperature at interior grid points."
    )

def main():
    optparser = optparse.OptionParser()
    add_stationary_opt(optparser)
    add_n_opt(optparser)
    add_timestep_opt(optparser)
    add_locstep_opt(optparser)
    add_ts_opt(optparser)
    add_te_opt(optparser)
    add_tinit_opt(optparser)
    opts, args = optparser.parse_args()

if __name__ == "__main__":
    main()
    
