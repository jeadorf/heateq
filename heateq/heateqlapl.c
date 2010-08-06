#include <Python.h>
#include <numpy/arrayobject.h>

PyObject *heateqlapl_apply(PyObject *self, PyObject *args) {
    const PyObject *tarr;
    const PyObject *txxarr;
    const PyObject *ttoparr, *tbottomarr, *tleftarr, *trightarr;
    double *t, *txx, *ttop, *tbottom, *tleft, *tright;
    npy_intp m, n;
    int i, j;

    if (!PyArg_ParseTuple(args, "OOOOOO", &tarr, &txxarr, &ttoparr, &tbottomarr, &tleftarr, &trightarr)) {
        return NULL;
    }

    m = PyArray_DIM(tarr, 0);
    n = PyArray_DIM(tarr, 1);

    t = PyArray_DATA(tarr);
    txx = PyArray_DATA(txxarr);
    ttop = PyArray_DATA(ttoparr);
    tbottom = PyArray_DATA(tbottomarr);
    tright = PyArray_DATA(trightarr);
    tleft = PyArray_DATA(tleftarr);

    /* further speedup is possible but it seems fast enough */
    txx[0] = ttop[0] + t[n] + tleft[0] + t[1] - 4 * t[0];
    for (j = 1; j < n - 1; j++) {
        txx[j] = ttop[j] + t[n+j] + t[j-1] + t[j+1] - 4*t[j];
    }
    txx[n-1] = ttop[n-1] + t[n+n-1] + t[n-2] + tright[0] - 4*t[n-1];
    for (i = 1; i < m - 1; i++) {
        txx[i*n] = t[(i-1)*n] + t[(i+1)*n] + tleft[i] + t[i*n+1] - 4*t[i*n];
        for (j = 1; j < n - 1; j++) {
           txx[i*n+j] = t[(i-1)*n+j] + t[(i+1)*n+j] + t[i*n+j-1] + t[i*n+j+1] - 4*t[i*n+j];
        }
        txx[i*n+n-1] = t[(i-1)*n+n-1] + t[(i+1)*n+n-1] + t[i*n+n-2] + tright[i] - 4*t[i*n+n-1];
    }
    txx[(m-1)*n] = t[(m-2)*n] + tbottom[0] + tleft[0] + t[(m-1)*n+1] - 4*t[(m-1)*n];
    for (j = 1; j < n - 1; j++) {
        txx[(m-1)*n+j] = t[(m-2)*n+j] + tbottom[j] + t[(m-1)*n+j-1] + t[(m-1)*n+j+1]- 4*t[(m-1)*n+j];
    }
    txx[(m-1)*n+n-1] = t[(m-2)*n+n-1] + tbottom[n-1] + t[(m-1)*n+n-2] + tright[m-1]- 4*t[(m-1)*n+n-1];

    return PyInt_FromLong(0);
}


static PyMethodDef HeateqLaplMethods[] = {
    {"apply",  heateqlapl_apply, METH_VARARGS, "Apply the 2d Laplace operator."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

PyMODINIT_FUNC initheateqlapl(void) {
    (void) Py_InitModule("heateqlapl", HeateqLaplMethods);
}

