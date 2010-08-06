from distutils.core import setup, Extension
import numpy

m1 = Extension('laplace2d',
                    sources = ['laplace2d.c'])
m2 = Extension('plot2d',
                    libraries=["cairo"],
                    sources = ['plot2d.c'])

setup (name = 'Heateq C extensions',
       version = '0.1',
       description = 'C extensions for heateq',
       include_dirs=[numpy.get_include(), '/usr/include/cairo'],
       ext_modules = [m1, m2])

