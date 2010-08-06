from distutils.core import setup, Extension
import numpy

module = Extension('laplace2d',
                    sources = ['laplace2d.c'])

setup (name = 'Laplace2d',
       version = '0.1',
       description = 'Applies the 2d Laplace operator using finite differences',
       include_dirs=[numpy.get_include()],
       ext_modules = [module])
