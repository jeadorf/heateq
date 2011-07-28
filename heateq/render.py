#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cairo
import gtk
import solver
import sys
import numpy as np
import heateqrndr

class RenderingContext:
    """Configures the rendering process of a heat distribution."""

    def __init__(self, tblue, tred, width=600, height=600, squares=False, interpolate=True):
        """Construct a rendering context.

        tblue,       -- Specify the temperature values for colors blue and red.
        tred            All temperature values less or equal than tblue are
                        painted in pure RGB blue and all values larger or equal
                        than tred are painted in pure RGB red.

        width,       -- Restrict the size of the rendered image.
        height         

        squares      -- Whether to draw grid cells as squares (ignored for 1D).
                        Setting this flag might lead that parts of the
                        specified canvas will not be drawn upon because
                        arguments 'width' and 'height' can only be interpreted
                        as upper bounds.

        interpolate  -- Enable interpolation.  If interpolation is not
                        supported by the renderer this flag will be ignored.
       """
        self.tblue = tblue
        self.tred = tred
        self.width = width
        self.height = height
        self.squares = squares
        self.interpolate = interpolate
    

class TPlot2d(gtk.DrawingArea):
    """A GTK drawing area that paints the distribution of heat over the
    simulated area.  The drawing routines are buffered.  Nevertheless,
    redrawing the simulated area is a time-consuming operation and should be
    kept to a minimum in order to keep the GUI application responsive.
    """
    
    def __init__(self, context):
        super(TPlot2d, self).__init__()
        self.connect("expose_event", self.expose)
        self.context = context
        self.oldsize = None 
        self.t = np.zeros((0,))
        self.c_buf = None
        self.buffer_cr = None

    def set_t(self, t):
        self.t = t
   
    def _must_recreate_buffers(self):
        if self.oldsize == None or self.c_buf == None or self.buffer_cr == None:
            return True
        elif self.t.shape != self.c_buf.shape:
            return True
        elif self.oldsize != (self.get_allocation().width, self.get_allocation().height):
            return True
        else:
            return False

    def expose(self, widg, evt):
        cr = widg.window.cairo_create()
        rect = self.get_allocation()
        if self._must_recreate_buffers():
            self.buffer_image = cairo.ImageSurface(cairo.FORMAT_ARGB32, rect.width, rect.height)
            self.buffer_cr = cairo.Context(self.buffer_image)
            self.oldsize = (rect.width, rect.height)
            self.c_buf = -np.ones(self.t.shape, dtype=np.double)
        if self.t != None:
            render2d(self.t, self.buffer_cr, rect.x, rect.y, rect.width, rect.height, self.context.tblue, self.context.tred, self.c_buf)
            cr.rectangle(rect.x, rect.y, rect.width, rect.height)
            cr.clip()
            cr.set_source_surface(self.buffer_image)
            cr.paint()
        return True


class TPlot1d(gtk.DrawingArea):
    """A GTK drawing area that paints the distribution of heat over the
    simulated pipe.
    """

    def __init__(self, context):
        super(TPlot1d, self).__init__()
        self.connect("expose_event", self.expose)
        self.context = context
        # todo: what ist the default array dtype in numpy ?
        self.t = np.zeros((1,))
    
    def expose(self, widg, evt):
        cr = widg.window.cairo_create()
        rect = self.get_allocation()
        render1d(self.t, cr, rect.x, rect.y, rect.width, rect.height, self.context.tblue, self.context.tred)
        return True

def render1d(t, ctx, x, y, w, h, tmin, tmax, interpolate=True):
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
render2d = heateqrndr.render2d

def render(t, cr, context):
    """Render heat distribution in the simulated area.

        t        -- Array of temperature values.  The shape tells the renderer whether
                    to render a pipe or a rectangle.

        cr      -- Cairo context.

        context -- An instance of class RenderingContext.

    """
    # TODO: unpack context in render1d, render2d methods
    ndims = len(t.shape)
    if ndims == 1:
        render1d(t, cr, 0, 0, context.width, context.height, context.tblue, context.tred, context.interpolate)
    elif ndims == 2:
        render2d(t, cr, 0, 0, context.width, context.height, context.tblue, context.tred)
    else:
        raise ValueError("Unsupported dimension: %d" % ic.dim)

def render_pdf(t, context, filename):
    pdf = cairo.PDFSurface(filename, context.width, context.height)
    cr = cairo.Context(pdf)
    ndims = len(t.shape)
    render(t, cr, context)
    pdf.flush()

def render_win(t):
    ndims = len(t.shape)
    if ndims == 1:
        _show_win_1d_stationary(t)    
    elif ndims == 2:
        _show_win_2d_stationary(t)    
    else:
        raise ValueError("Unsupported dimension: %d" % ic.dim)

def _show_win_1d_stationary(t):
    win = gtk.Window()
    win.set_title("Temperature curve, n=%d" % len(t))
    win.set_default_size(800, 100)
    context = RenderingContext(t.min(), t.max())
    plot = TPlot1d(context)
    plot.t = t
    win.add(plot)
    win.connect("destroy", gtk.main_quit)
    win.show_all()
    gtk.main()

def _show_win_2d_stationary(t):
    win = gtk.Window()
    m, n = len(t), len(t[0])
    win.set_title("Temperature curve, m=%d, n=%d" % (m, n))
    win.set_default_size(800, 800)
    context = RenderingContext(t.min(), t.max())
    plot = TPlot2d(context)
    plot.t = t
    win.add(plot)
    win.connect("destroy", gtk.main_quit)
    win.show_all()
    gtk.main()

