#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cairo
import gtk
import solver
import sys
import numpy as np
import heateqplot

class TPlot2d(gtk.DrawingArea):
    
    def __init__(self, m, n, tmin, tmax):
        super(TPlot2d, self).__init__()
        self.connect("expose_event", self.expose)
        self.shape = (m, n)
        self.tmin = tmin
        self.tmax = tmax
        self.oldsize = None 
        self.t = None

    def set_t(self, t):
        if t.shape == self.shape:
            self.t = t
        else:
            raise ValueError("Shape of temperature value array does not fit")
    
    def expose(self, widg, evt):
        cr = widg.window.cairo_create()
        rect = self.get_allocation()
        newsize = (rect.width, rect.height)
        if self.oldsize == None or self.oldsize != newsize:
            self.buffer_image = cairo.ImageSurface(cairo.FORMAT_ARGB32, newsize[0], newsize[1])
            self.buffer_cr = cairo.Context(self.buffer_image)
            self.oldsize = newsize
            self.c_buf = -np.ones(self.shape, dtype=np.double)
        if self.t != None:
            plot2d(self.t, self.buffer_cr, rect.x, rect.y, rect.width, rect.height, self.tmin, self.tmax, self.c_buf)
            cr.rectangle(rect.x, rect.y, rect.width, rect.height)
            cr.clip()
            cr.set_source_surface(self.buffer_image)
            cr.paint()
        return True

class TPlot1d(gtk.DrawingArea):

    def __init__(self, tmin, tmax):
        super(TPlot1d, self).__init__()
        self.connect("expose_event", self.expose)
        self.tmin = tmin
        self.tmax = tmax
        # todo: what ist the default array dtype in numpy ?
        self.t = np.zeros((1,))
    
    def expose(self, widg, evt):
        cr = widg.window.cairo_create()
        rect = self.get_allocation()
        plot1d(self.t, cr, rect.x, rect.y, rect.width, rect.height, self.tmin, self.tmax)
        return True

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
    plot = TPlot1d(t.min(), t.max())
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
    plot = TPlot2d(m, n, tmin, tmax)
    plot.t = t
    win.add(plot)
    win.connect("destroy", gtk.main_quit)
    win.show_all()
    gtk.main()

if __name__ == "__main__":
    main()

