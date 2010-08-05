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
    def __init__(self):
        super(TemperaturePlot, self).__init__()
        self.connect("expose_event", self.expose)
        self.t = [0, 1]
    def expose(self, widget, evt):
        ctx = widget.window.cairo_create()
        ctx.rectangle(evt.area.x, evt.area.y, evt.area.width, evt.area.height)
        ctx.clip()
        self.redraw(ctx)
        return False
    def redraw(self, ctx):
        ctx = self.window.cairo_create()
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
    ts, te, n = 5, -5, 300
    t = sol.solve_stationary_1d(ts, te, n)
    gen_pdf_1d(t, "waermeleitung.pdf")
    # show_win_1d_stationary(t)
    show_win_1d()

def gen_pdf_1d(t, filename):
    pdf = cairo.PDFSurface(filename, 600, 100)
    ctx = cairo.Context(pdf)
    plot_1d(t, ctx, 10, 10, 580, 80)
    pdf.flush()

def show_win_1d_stationary(t):
    win = gtk.Window()
    win.set_title("Temperature curve, n=%d" % (len(t)-2))
    win.set_default_size(800, 100)
    plot = TemperaturePlot()
    plot.t = t
    win.add(plot)
    win.connect("destroy", gtk.main_quit)
    win.show_all()
    gtk.main()

import sys, gobject

def show_win_1d():
    win = gtk.Window()
    win.set_default_size(800, 100)
    plot = TemperaturePlot()
    win.add(plot)
    win.connect("destroy", gtk.main_quit)
    win.show_all()

    def update(t):
        plot.t = t
        plot.redraw(None)

    br = False
    class SimThread(threading.Thread):
        def run(self):
            ts = 5
            te = 0
            t_init = [0 for i in xrange(0, 10)]
            i = 0
            tc = []
            for t, time in sol.sim_heateq_1d(ts, te, t_init, 1500, 5, 0.001):
                if i % 3 == 0:
                    gobject.idle_add(update, copy.copy(t))
                    time.sleep(1./30)
                i += 1
                if br:
                    break

    gtk.gdk.threads_init()
    simthread = SimThread()
    simthread.start()

    gtk.main()
    br = True

if __name__ == "__main__":
    main()

