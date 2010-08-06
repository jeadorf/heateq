#include <Python.h>
#include <stdio.h>
#include <numpy/arrayobject.h>

#define C_ORDER_POS(i, j) ((i) * n + (j))

PyObject *laplace2d_apply(PyObject *self, PyObject *args) {
    // Contains the vector of temperatur values */
    const PyArrayObject *tarr;
    // Collecting parameter that will contain approximations for the sum of the second-order 
    // derivatives.
    const PyArrayObject *txxarr;
    // Boundary conditions
    const PyArrayObject *ttoparr, *tbottomarr, *tleftarr, *trightarr;
    // Pointers to raw array data
    double *t, *txx, *ttop, *tbottom, *tleft, *tright;
    // Array dimensions
    npy_intp m, n;
    // Loop variables 
    int i, j;

    if (!PyArg_ParseTuple(args, "OOOOOO", &tarr, &txxarr, &ttoparr, &tbottomarr, &tleftarr, &trightarr)) {
        return NULL;
    }

    m = tarr->dimensions[0];
    n = tarr->dimensions[1];

    /* todo: use macros */
    t = (double*) tarr->data;
    txx = (double*) txxarr->data;
    ttop = (double*) ttoparr->data;
    tbottom = (double*) tbottomarr->data;
    tright = (double*) trightarr->data;
    tleft = (double*) tleftarr->data;

/*
    printf("m=%d,n=%d,nd=%d,len=%d,elsize=%d\n", m, n, tarr->nd, m*n, tarr->descr->elsize);
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            printf("%.2f ", t[i * n + j]);
        }
        puts("\n");
    }*/

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
/*
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            printf("%.2f ", txx[i * n + j]);
        }
        puts("\n");
    }*/

    return PyInt_FromLong((long) 0);
}


static PyMethodDef Laplace2dMethods[] = {
    {"apply",  laplace2d_apply, METH_VARARGS, "Apply the 2d Laplace operator."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

PyMODINIT_FUNC initlaplace2d(void) {
    (void) Py_InitModule("laplace2d", Laplace2dMethods);
}

