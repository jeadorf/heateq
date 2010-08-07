#include <Python.h>
#include <numpy/arrayobject.h>
#include <pycairo/pycairo.h>
#include <cairo/cairo.h>
#include <stdio.h>

static Pycairo_CAPI_t *Pycairo_CAPI;

PyObject *heateqplot_plot2d(PyObject *self, PyObject *args) {
    PyObject *t_arr;
    PyObject *c_buf_arr = NULL;
    double *t, *c_buf = NULL;
    const PyObject *crobj;
    double x, y, w, h, tmin, tmax, tspan;
    double c, eps, pxw, pxh;
    cairo_t *cr; 
    npy_intp m, n;
    int i, j;

    if (!PyArg_ParseTuple(args, "OOdddddd|O", &t_arr, &crobj, &x, &y, &w, &h, &tmin, &tmax, &c_buf_arr)) {
        return NULL;
    }

    m = PyArray_DIM(t_arr, 0);
    n = PyArray_DIM(t_arr, 1);

    t = PyArray_DATA(t_arr);
    if (c_buf_arr != NULL) {
        c_buf = PyArray_DATA(c_buf_arr);
    }
    cr = PycairoContext_GET(crobj);

    cairo_save(cr);
    cairo_translate(cr, x, y);
    cairo_scale(cr, 1. * w / n, 1. * h / m);
    // pixel correcture to avoid artifacts due to rounding errors
    pxw = .5 * n / w;
    pxh = .5 * m / h;
    tspan = tmax - tmin;
    c;
    if (tspan == 0) {
        tspan = 1;
    }
    cairo_rectangle(cr, 0, 0, 1 + 2*pxw, 1 + 2*pxh);
    cairo_path_t *r0 = cairo_copy_path(cr);
    cairo_new_path(cr);
    int skip_flag = 0;
    for (i = 0; i < m; i++) {
        cairo_save(cr);
        for (j = 0; j < n; j++) {
            c = 1. * (t[i*n+j] - tmin) / tspan;
            // Redraw this patch only if it is not already painted in a very
            // similar color.  Retain the color value in a per-plot buffer that
            // is provided by the caller.  The buffer must not be shared
            // between different cairo contexts.
            if (c_buf != NULL) {
                eps = c_buf[i*n+j] - c; 
                if (eps < 0.01 && eps > -0.01) {
                    skip_flag = 1;
                } else {
                    skip_flag = 0;
                }
            }
            if (skip_flag) {
                // cairo_set_source_rgb(cr, 0, 0, 0);
            } else {
                cairo_append_path(cr, r0);
                cairo_set_source_rgb(cr, c, 0, 1 - c);
                cairo_fill(cr);
                if (c_buf != NULL) {
                    c_buf[i*n+j] = c;
                }
            }
            cairo_translate(cr, 1, 0);
        }
        cairo_restore(cr);
        cairo_translate(cr, 0, 1);
    }
    cairo_restore(cr);
    cairo_path_destroy(r0);

    return PyInt_FromLong(0L);
}


static PyMethodDef HeateqPlotMethods[] = {
    {"plot2d",  heateqplot_plot2d, METH_VARARGS, "Plot a 2d temperature grid."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

PyMODINIT_FUNC initheateqplot(void) {
    Pycairo_IMPORT;
    (void) Py_InitModule("heateqplot", HeateqPlotMethods);
}

