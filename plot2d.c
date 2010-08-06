#include <Python.h>
#include <numpy/arrayobject.h>
#include <pycairo/pycairo.h>
#include <stdio.h>

static Pycairo_CAPI_t *Pycairo_CAPI;

// plot2d.plot2d(t, ctx, x, y, w, h, tmin, tmax) 
PyObject *plot2d_plot2d(PyObject *self, PyObject *args) {
    // Contains the vector of temperatur values
    const PyObject *tarr;
    // Raw temperature data 
    double *t;
    // Contains the drawing context
    const PyObject *ctxobj;
    double x, y, w, h, tmin, tmax;
    cairo_t *ctx; 
    // Array dimensions
    npy_intp m, n;
    // Loop variables 
    int i, j;

    if (!PyArg_ParseTuple(args, "OOdddddd", &tarr, &ctxobj, &x, &y, &w, &h, &tmin, &tmax)) {
        return NULL;
    }

    puts("hello from plot2d\n");
    printf("x=%f,y=%f,w=%f,h=%f", x, y, w, h);

    m = PyArray_DIM(tarr, 0);
    n = PyArray_DIM(tarr, 1);

    t = PyArray_DATA(tarr);
    ctx = PycairoContext_GET(ctxobj);

    //cairo_rectangle(ctx, x, y, w, h);
    //cairo_set_source_rgb(ctx, 1.0, 0.0, 0.0);
    //cairo_fill(ctx);

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

