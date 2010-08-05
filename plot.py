#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cairo
import gtk
import sol
import math

class PlotCanvas(gtk.DrawingArea):
    def __init__(self):
        super(PlotCanvas, self).__init__()
        self.connect("expose_event", self.expose)
        self.t = [0, 1]
    def expose(self, widget, evt):
        ctx = widget.window.cairo_create()
        ctx.rectangle(evt.area.x, evt.area.y, evt.area.width, evt.area.height)
        ctx.clip()
        self.redraw(ctx)
        return False
    def redraw(self, ctx):
        rect = self.get_allocation()
        plot_1d(self.t, ctx, rect.x, rect.y, rect.width, rect.height)  

def plot_1d(t, ctx, x, y, w, h):
    ctx.save()
    ctx.translate(x, y)
    n = len(t)
    tmin = min(t)
    tmax = max(t)
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

def main():
    pdf = cairo.PDFSurface("waermeleitung.pdf", 600, 100)
    ctx = cairo.Context(pdf)
    ts, te, n = 5, -5, 300
    t = sol.solve_stationary_1d(ts, te, n)
    plot_1d(t, ctx, 10, 10, 580, 80)
    pdf.flush()

    win = gtk.Window()
    win.set_title("Temperature curve, te=%.2f, ts=%.2f, n=%d" % (ts, te, n))
    win.set_default_size(800, 100)
    plot = PlotCanvas()
    plot.t = t
    win.add(plot)
    win.connect("destroy", gtk.main_quit)
    win.show_all()
    gtk.main()

if __name__ == "__main__":
    main()

