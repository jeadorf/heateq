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
        return True
    def redraw(self, ctx=None):
        if ctx == None:
            ctx = self.window.cairo_create()
        rect = self.get_allocation()
        plot_1d(self.t, ctx, rect.x, rect.y, rect.width, rect.height, self.tmin, self.tmax)

def plot_1d(t, ctx, x, y, w, h, tmin, tmax, interpolate=True):
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

def plot_2d(t, ctx, x, y, w, h, tmin, tmax):
    ctx.save()
    m = len(t)
    n = len(t[0])
    ctx.translate(x, y)
    ctx.scale(1. * w / n, 1. * h / m)
    # pixel correcture to avoid artifacts due to rounding errors
    pxw = .5 * n / w
    pxh = .5 * m / h
    tspan = tmax - tmin
    if tspan == 0:
        tspan = 1
    for i in xrange(0, m):
        ctx.save()
        for j in xrange(0, n):
            c = 1. * (t[i][j] - tmin) / tspan
            if i > 0 and j > 0 and i < m - 1 and j < n - 1:
                #
                c1 = 1. * (t[i-1][j+1] - tmin) / tspan
                c2 = 1. * (t[i-1][j-1] - tmin) / tspan
                c3 = 1. * (t[i+1][j-1] - tmin) / tspan
                c4 = 1. * (t[i+1][j+1] - tmin) / tspan
                #
                g1 = cairo.LinearGradient(0, 0, 1, -1)
                g2 = cairo.LinearGradient(0, 0, -1, -1)
                g3 = cairo.LinearGradient(0, 0, -1, 1)
                g4 = cairo.LinearGradient(1, 1, 0, 0)
                #
                g1.add_color_stop_rgb(0, c, 0, 1 - c)
                g2.add_color_stop_rgb(0, c, 0, 1 - c)
                g3.add_color_stop_rgb(0, c, 0, 1 - c)
                g4.add_color_stop_rgb(0, c, 0, 1 - c)
                #
                g1.add_color_stop_rgb(1, c1, 0, 1 - c1)
                g2.add_color_stop_rgb(1, c2, 0, 1 - c2)
                g3.add_color_stop_rgb(1, c3, 0, 1 - c3)
                g4.add_color_stop_rgb(1, c4, 0, 1 - c4)
                #
                ctx.set_source_rgb(c, 0, 1 - c)
                ctx.rectangle(0.5, 0, 0.5+pxw, 0.5+pxh)
                #ctx.set_source_rgb(0, 0, 1)
                ctx.set_source(g1)
                ctx.fill()
                ctx.set_source_rgb(c, 0, 1 - c)
                ctx.rectangle(0.0, 0.0, 0.5+pxw, 0.5+pxh)
                #ctx.set_source_rgb(0, 1, 0)
                ctx.set_source(g2)
                ctx.fill()
                ctx.rectangle(0.0, 0.5, 0.5+pxw, 0.5+pxh)
                #ctx.set_source_rgb(1, 0, 0)
                ctx.set_source(g3)
                ctx.fill()
                ctx.rectangle(0.5, 0.5, 0.5+pxw, 0.5+pxh)
                #ctx.set_source_rgb(1, 0, 1)
                ctx.set_source(g4)
                #ctx.set_source_rgb(c, 0, 1 - c)
                ctx.fill()
            else:
                ctx.rectangle(0, 0, 1 + pxw, 1 + pxh)
                ctx.set_source_rgb(c, 0, 1 - c)
                ctx.fill()
            ctx.translate(1, 0)
        ctx.restore()
        ctx.translate(0, 1)
    ctx.restore()


def gen_pdf_1d(t, filename):
    pdf = cairo.PDFSurface(filename, 600, 100)
    ctx = cairo.Context(pdf)
    plot_1d(t, ctx, 10, 10, 580, 80, min(t), max(t))
    pdf.flush()

def gen_pdf_2d(t, filename):
    pdf = cairo.PDFSurface(filename, 300, 300)
    ctx = cairo.Context(pdf)
    plot_2d(t, ctx, 10, 10, 280, 280, 0, 1)
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

