#include <Python.h>
#include <numpy/arrayobject.h>
#include <pycairo/pycairo.h>
#include <cairo/cairo.h>
#include <stdio.h>

static Pycairo_CAPI_t *Pycairo_CAPI;

/** 
 * Render 2-dimensional heat distribution.
 *
 * @param t_arr Temperature numpy array.
 * @param crobj Cairo context.
 * @param x     x-offset in px.
 * @param y     y-offset in px.
 * @param w     Width in px.
 * @param h     Height in px.
 * @param tblue    See RenderingContext.
 * @param tred     See RenderingContext. 
 * @param c_buf_arr     Provide a color buffer that might speed up the
 *      rendering process in certain cases.  Use the same color buffer when
 *      rendering heat distributions on the same surface and only if you know
 *      that the surface does not "lose" its paint (which allows to selectively
 *      repaint only those tiles that changed its color significantly).
 *      (optional)
 * @param skip_thresh   If the change of color in a tile is larger than
 *      skip_thresh the tile will be redrawn.  (optional)
 */
PyObject *heateqrndr_render2d(PyObject *self, PyObject *args) {
    PyObject *t_arr;
    PyObject *c_buf_arr = NULL;
    double *t, *c_buf = NULL;
    const PyObject *crobj;
    double x, y, w, h, tmin, tmax;
    double skip_thresh = 0.01;
    double c, eps, pxw, pxh, tspan;
    cairo_t *cr; 
    npy_intp m, n;
    int i, j;

    if (!PyArg_ParseTuple(args, "OOdddddd|Od", &t_arr, &crobj, &x, &y, &w, &h, &tmin, &tmax, &c_buf_arr, &skip_thresh)) {
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
    pxw = .25 * n / w;
    pxh = .25 * m / h;
    tspan = tmax - tmin;
    if (tspan == 0) {
        tspan = 1;
    }
    cairo_rectangle(cr, 0, 0, 1 + 2*pxw, 1 + 2*pxh);
    cairo_path_t *r0 = cairo_copy_path(cr);
    cairo_new_path(cr);
    int skip = 0;
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
                if (eps < skip_thresh && eps > skip_thresh) {
                    skip = 1;
                } else {
                    skip = 0;
                    c_buf[i*n+j] = c;
                }
            }
            if (!skip) {
                cairo_append_path(cr, r0);
                cairo_set_source_rgb(cr, c, 0, 1 - c);
                // Considering the profiler statistics this call to cairo_fill
                // is time-consuming and calls should be avoided if possible. 
                cairo_fill(cr);
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


static PyMethodDef HeateqRenderMethods[] = {
    {"render2d",  heateqrndr_render2d, METH_VARARGS, "Plot a 2d temperature grid."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

PyMODINIT_FUNC initheateqrndr(void) {
    Pycairo_IMPORT;
    (void) Py_InitModule("heateqrndr", HeateqRenderMethods);
}

