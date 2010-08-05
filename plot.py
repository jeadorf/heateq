#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cairo
import gtk
import sol
import math
import copy
import time
import threading

class TemperaturePlot(gtk.DrawingArea):
    def __init__(self, tmin, tmax):
        super(TemperaturePlot, self).__init__()
        self.connect("expose_event", self.expose)
        self.t = [0, 1]
        self.tmin = tmin
        self.tmax = tmax
    def expose(self, widget, evt):
        ctx = widget.window.cairo_create()
        ctx.rectangle(evt.area.x, evt.area.y, evt.area.width, evt.area.height)
        ctx.clip()
        self.redraw(ctx)
        return False
    def redraw(self, ctx):
        ctx = self.window.cairo_create()
        rect = self.get_allocation()
        plot_1d(self.t, ctx, rect.x, rect.y, rect.width, rect.height, self.tmin, self.tmax)

def plot_1d(t, ctx, x, y, w, h, tmin, tmax):
    ctx.save()
    ctx.translate(x, y)
    n = len(t)
    tspan = tmax - tmin
    if tspan == 0:
        tspan = 1
    for i in xrange(0, n):
        ctx.rectangle(0, 0, 1. * w/n + 1, h)
        l = 1. * (t[i] - tmin) / tspan
        ctx.set_source_rgb(l, 0, 1 - l)
        ctx.fill()
        ctx.translate(1. * w/n, 0)
    ctx.restore()

def gen_pdf_1d(t, filename):
    pdf = cairo.PDFSurface(filename, 600, 100)
    ctx = cairo.Context(pdf)
    plot_1d(t, ctx, 10, 10, 580, 80, min(t), max(t))
    pdf.flush()

def show_win_1d_stationary(t):
    win = gtk.Window()
    win.set_title("Temperature curve, n=%d" % (len(t)-2))
    win.set_default_size(800, 100)
    plot = TemperaturePlot(min(t), max(t))
    plot.t = t
    win.add(plot)
    win.connect("destroy", gtk.main_quit)
    win.show_all()
    gtk.main()

if __name__ == "__main__":
    main()

