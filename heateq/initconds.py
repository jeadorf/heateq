#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class InitConds(object):
    """Abstract class for initial values and boundary conditions.  The examined
    heat diffusion problems are completely determined by the initial values at
    the interior of the simulated area (time 0) and a function for each
    boundary that returns a vector of temperature values at this boundary for
    any given point of time.  
    """

    def __init__(self, dim):
        self.dim = dim


class InitConds1d(InitConds):
    """Describes the initial values and boundary conditions of a 1-dimensional
    pipe. See InitConds for a more detailed description.
    """

    def __init__(self, n, left=None, right=None, interior=None):
        super(InitConds1d, self).__init__(1)
        self.n = n
        if right == None:
            self.right = const(0)
        else:
            self.right = right
        if left == None:
            self.left = const(0)
        else:
            self.left = left
        if interior == None:
            self.interior = np.zeros((n,))
        else:
            self.interior = interior


class InitConds2d(InitConds):
    """Describes the initial values and boundary conditions of a 1-dimensional
    pipe. See InitConds for a more detailed description.
    """
    
    def __init__(self, m, n, top=None, right=None, bottom=None, left=None, interior=None):
        super(InitConds2d, self).__init__(2)
        self.m = m
        self.n = n
        if top == None:
            self.top = const_zeros(n)
        else:
            self.top = top
        if bottom == None:
            self.bottom = const_zeros(n)
        else:
            self.bottom = bottom 
        if right == None:
            self.right = const_zeros(n,)
        else:
            self.right = right
        if left == None:
            self.left = const_zeros(n)
        else:
            self.left = left
        if interior == None:
            self.interior = np.zeros((m, n))
        else:
            self.interior = interior
    
    def shape(self):
        return self.m, self.n


def const(val):
    """Construct constant boundary values."""
    return (lambda tm: val)

def const_zeros(*shape):
    """Construct constant zero boundary values.  This will set the temperature
    at the complete boundary to 0.
    """
    val = np.zeros(shape)
    return (lambda tm: val)

def const_ones(*shape):
    """Construct constant zero boundary values.  This will set the temperature
    at the complete boundary to 1.
    """
    val = np.ones(shape)
    return (lambda tm: val)
