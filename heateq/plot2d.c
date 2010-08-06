#include <Python.h>
#include <numpy/arrayobject.h>
#include <pycairo/pycairo.h>
#include <cairo/cairo.h>
#include <stdio.h>

static Pycairo_CAPI_t *Pycairo_CAPI;

PyObject *heateqplot_plot2d(PyObject *self, PyObject *args) {
    const PyObject *tarr;
    double *t;
    const PyObject *crobj;
    double x, y, w, h, tmin, tmax;
    cairo_t *cr; 
    npy_intp m, n;
    int i, j;

    if (!PyArg_ParseTuple(args, "OOdddddd", &tarr, &crobj, &x, &y, &w, &h, &tmin, &tmax)) {
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
    double c;
    if (tspan == 0) {
        tspan = 1;
    }
    cairo_rectangle(cr, 0, 0, 1 + 2*pxw, 1 + 2*pxh);
    cairo_path_t *r0 = cairo_copy_path(cr);
    cairo_new_path(cr);
    for (i = 0; i < m; i++) {
        cairo_save(cr);
        for (j = 0; j < n; j++) {
            c = 1. * (t[i*n+j] - tmin) / tspan;
            cairo_append_path(cr, r0);
            cairo_set_source_rgb(cr, c, 0, 1 - c);
            cairo_fill(cr);
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

