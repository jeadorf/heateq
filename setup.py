import setuptools
from distutils.core import Extension
import numpy

m1 = Extension('heateqlapl',
                    sources = ['heateq/heateqlapl.c'])
m2 = Extension('heateqplot',
                    libraries=["cairo"],
                    sources = ['heateq/heateqplot.c'])

setuptools.setup(name = 'Heateq',
      author='Julius Adorf',
      version = '0.1',
      description = 'Heat equation simulator',
      include_dirs=[numpy.get_include(), '/usr/include/cairo'],
      ext_modules = [m1, m2],
      packages = setuptools.find_packages(),
      platforms = 'any',
      license='BSD License',
      keywords='heat equation')
