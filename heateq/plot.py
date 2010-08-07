#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cairo
import gtk
import solver
import sys
import numpy as np
import heateqplot

class TemperaturePlot(gtk.DrawingArea):

    def __init__(self, m, n, tmin, tmax, dim=1):
        super(TemperaturePlot, self).__init__()
        self.connect("expose_event", self.expose)
        self.m = m
        self.n = n
        self.tmin = tmin
        self.tmax = tmax
        self.dim = dim
        if m > 0:
            self.t = np.zeros((1, 1))
            self.c_buf = np.zeros((1, 1), dtype=np.double) 
        else:
            self.t = np.zeros((1,))

    def expose(self, widget, evt):
        ctx = widget.window.cairo_create()
        ctx.rectangle(evt.area.x, evt.area.y, evt.area.width, evt.area.height)
        ctx.clip()
        self.redraw(ctx)
        return True

    def redraw(self, ctx=None):
        if ctx == None:
            ctx = self.window.cairo_create()
        rect = self.get_allocation()
        if self.dim == 2:
            if self.c_buf.shape != self.t.shape:
                self.c_buf = -1. * np.ones(self.t.shape, dtype=np.double)
            plot2d(self.t, ctx, rect.x, rect.y, rect.width, rect.height, self.tmin, self.tmax)
        else:
            plot1d(self.t, ctx, rect.x, rect.y, rect.width, rect.height, self.tmin, self.tmax)


def plot1d(t, ctx, x, y, w, h, tmin, tmax, interpolate=True):
    ctx.save()
    n = len(t)
    ctx.translate(x, y)
    ctx.scale(1. * w / n, h)
    # pixel correcture to avoid artifacts due to rounding errors
    px = 1. * n / w
    tspan = tmax - tmin
    if tspan == 0:
        tspan = 1
    cc = 0
    for i in xrange(0, n):
        ctx.rectangle(0, 0, 1 + px, 1)
        c = 1. * (t[i] - tmin) / tspan
        if interpolate:
            grad = cairo.LinearGradient(0, 0, 1, 0)
            if i > 0:
                grad.add_color_stop_rgb(0, cc, 0, 1 - cc)
            grad.add_color_stop_rgb(1, c, 0, 1 - c)
            cc = c
            ctx.set_source(grad)
        else:
            ctx.set_source_rgb(c, 0, 1 - c)
        ctx.fill()
        ctx.translate(1, 0)
    ctx.restore()

# Expose C extension
plot2d = heateqplot.plot2d

def gen_pdf_1d(t, filename):
    pdf = cairo.PDFSurface(filename, 600, 100)
    ctx = cairo.Context(pdf)
    plot1d(t, ctx, 10, 10, 580, 80, min(t), max(t))
    pdf.flush()

def gen_pdf_2d(t, filename):
    pdf = cairo.PDFSurface(filename, 600, 600)
    ctx = cairo.Context(pdf)
    plot2d(t, ctx, 10, 10, 580, 580, 0, 1)
    pdf.flush()

def show_win_1d_stationary(t):
    win = gtk.Window()
    win.set_title("Temperature curve, n=%d" % (len(t)-2))
    win.set_default_size(800, 100)
    plot = TemperaturePlot(len(t), t.min(), t.max())
    plot.t = t
    win.add(plot)
    win.connect("destroy", gtk.main_quit)
    win.show_all()
    gtk.main()

def show_win_2d_stationary(t):
    win = gtk.Window()
    m, n = len(t), len(t[0])
    win.set_title("Temperature curve, m=%d, n=%d" % (m, n))
    win.set_default_size(800, 800)
    tmin = sys.maxint
    tmax = -sys.maxint
    tmin = min(tmin, t.min())
    tmax = max(tmax, t.max())
    plot = TemperaturePlot(m, n, tmin, tmax, dim=2)
    plot.t = t
    win.add(plot)
    win.connect("destroy", gtk.main_quit)
    win.show_all()
    gtk.main()

if __name__ == "__main__":
    main()

