#include <Python.h>
#include <numpy/arrayobject.h>

PyObject *heateqlapl_apply(PyObject *self, PyObject *args) {
    const PyObject *tarr;
    const PyObject *txxarr;
    const PyObject *ttoparr, *tbottomarr, *tleftarr, *trightarr;
    double *t, *txx, *ttop, *tbottom, *tleft, *tright;
    npy_intp m, n;
    int i, j, k;

    if (!PyArg_ParseTuple(args, "OOOOOO", &tarr, &txxarr, &ttoparr, &tbottomarr, &tleftarr, &trightarr)) {
        return NULL;
    }

    m = PyArray_DIM(tarr, 0);
    n = PyArray_DIM(tarr, 1);
    i = m*n;

    t = PyArray_DATA(tarr);
    txx = PyArray_DATA(txxarr);
    ttop = PyArray_DATA(ttoparr);
    tbottom = PyArray_DATA(tbottomarr);
    tright = PyArray_DATA(trightarr);
    tleft = PyArray_DATA(tleftarr);

    txx[0] = ttop[0] + t[n] + tleft[0] + t[1] - 4 * t[0];
    for (k = 1; k < n - 1; k++) {
        txx[k] = ttop[k] + t[k+n] + t[k-1] + t[k+1] - 4*t[k];
    }
    txx[k] = ttop[k] + t[k+n] + t[k-1] + tright[0] - 4*t[k];
    for (k = n; k < i - n; k++) {
        txx[k] = t[k-n] + t[k+n] + tleft[k/n-1] + t[k+1] - 4*t[k];
        for (k += 1, j = k + n - 2; k < j; k++) {
           txx[k] = t[k-n] + t[k+n] + t[k-1] + t[k+1] - 4*t[k];
        }
        txx[k] = t[k-n] + t[k+n] + t[k-1] + tright[k/n-1] - 4*t[k];
    }
    txx[k] = t[k-n] + tbottom[0] + tleft[0] + t[k+1] - 4*t[k];
    for (k += 1; k < i; k++) {
        txx[k] = t[k-n] + tbottom[k%n] + t[k-1] + t[k+1]- 4*t[k];
    }
    txx[k] = t[k-n] + tbottom[n-1] + t[k-1] + tright[m-1]- 4*t[k];

    return PyInt_FromLong(0);
}


static PyMethodDef HeateqLaplMethods[] = {
    {"apply",  heateqlapl_apply, METH_VARARGS, "Apply the 2d Laplace operator."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

PyMODINIT_FUNC initheateqlapl(void) {
    (void) Py_InitModule("heateqlapl", HeateqLaplMethods);
}

