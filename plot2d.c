#include <Python.h>
#include <numpy/arrayobject.h>
#include <pycairo/pycairo.h>
#include <cairo/cairo.h>
#include <stdio.h>

static Pycairo_CAPI_t *Pycairo_CAPI;

// UNUSED
PyObject *plot2d_plot2d(PyObject *self, PyObject *args) {
    const PyObject *tarr;
    double *t;
    const PyObject *crobj;
    double x, y, w, h, tmin, tmax;
    int interpolate = 1;
    cairo_t *cr; 
    npy_intp m, n;
    int i, j;

    if (!PyArg_ParseTuple(args, "OOdddddd|iO", &tarr, &crobj, &x, &y, &w, &h, &tmin, &tmax, &interpolate)) {
        return NULL;
    }

    m = PyArray_DIM(tarr, 0);
    n = PyArray_DIM(tarr, 1);

    t = PyArray_DATA(tarr);
    cr = PycairoContext_GET(crobj);

    cairo_save(cr);
    cairo_translate(cr, x, y);
    cairo_scale(cr, 1. * w / n, 1. * h / m);
    // pixel correcture to avoid artifacts due to rounding errors
    double pxw = .5 * n / w;
    double pxh = .5 * m / h;
    double tspan = tmax - tmin;
    double c, c1, c2, c3, c4;
    cairo_pattern_t *g1, *g2, *g3, *g4;
    if (tspan == 0) {
        tspan = 1;
    }
    for (i = 0; i < m; i++) {
        cairo_save(cr);
        for (j = 0; j < n; j++) {
            c = 1. * (t[i*n+j] - tmin) / tspan;
            cairo_set_source_rgb(cr, c, 0, 1 - c);
            cairo_rectangle(cr, 0, 0, 1 + 2*pxw, 1 + 2*pxh);
            cairo_fill(cr);
            if (interpolate) {
                if (i > 0 && j < n - 1) {
                    c1 = 1. * (t[(i-1)*n+j+1] - tmin) / tspan;
                    g1 = cairo_pattern_create_linear(0, 0, 1, -1);
                    cairo_pattern_add_color_stop_rgb(g1, 0, c, 0, 1 - c);
                    cairo_pattern_add_color_stop_rgb(g1, 1, c1, 0, 1 - c1);
                    cairo_rectangle(cr, 0.5, 0, 0.5+pxw, 0.5+pxh);
                    cairo_set_source(cr, g1);
                    cairo_fill(cr);
                    cairo_pattern_destroy(g1);
                }
                if (i > 0 && j > 0) {
                    c2 = 1. * (t[(i-1)*n+j-1] - tmin) / tspan;
                    g2 = cairo_pattern_create_linear(0, 0, -1, -1);
                    cairo_pattern_add_color_stop_rgb(g2, 0, c, 0, 1 - c);
                    cairo_pattern_add_color_stop_rgb(g2, 1, c2, 0, 1 - c2);
                    cairo_rectangle(cr, 0.0, 0.0, 0.5+pxw, 0.5+pxh);
                    cairo_set_source(cr, g2);
                    cairo_fill(cr);
                    cairo_pattern_destroy(g2);
                }
                if (i < m -1 && j > 0) {
                    c3 = 1. * (t[(i+1)*n+j-1] - tmin) / tspan;
                    g3 = cairo_pattern_create_linear(0, 0, -1, 1);
                    cairo_pattern_add_color_stop_rgb(g3, 0, c, 0, 1 - c);
                    cairo_pattern_add_color_stop_rgb(g3, 1, c3, 0, 1 - c3);
                    cairo_rectangle(cr, 0.0, 0.5, 0.5+pxw, 0.5+pxh);
                    cairo_set_source(cr, g3);
                    cairo_fill(cr);
                    cairo_pattern_destroy(g3);
                }
                if (i < m -1 && j < n - 1) {
                    c4 = 1. * (t[(i+1)*n+j+1] - tmin) / tspan;
                    g4 = cairo_pattern_create_linear(0, 0, 1, 1);
                    cairo_pattern_add_color_stop_rgb(g4, 0, c, 0, 1 - c);
                    cairo_pattern_add_color_stop_rgb(g4, 1, c4, 0, 1 - c4);
                    cairo_rectangle(cr, 0.5, 0.5, 0.5+pxw, 0.5+pxh);
                    cairo_set_source(cr, g4);
                    cairo_fill(cr);
                    cairo_pattern_destroy(g4);
                }
            }
            cairo_translate(cr, 1, 0);
        }
        cairo_restore(cr);
        cairo_translate(cr, 0, 1);
    }
    cairo_restore(cr);

    return PyInt_FromLong(0L);
}


static PyMethodDef Plot2dMethods[] = {
    {"plot2d",  plot2d_plot2d, METH_VARARGS, "Plot a 2d temperature grid."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

PyMODINIT_FUNC initplot2d(void) {
    Pycairo_IMPORT;
    (void) Py_InitModule("plot2d", Plot2dMethods);
}

