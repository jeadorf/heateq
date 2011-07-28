import setuptools
from distutils.core import Extension
import numpy

m1 = Extension('heateqlapl',
                    sources = ['heateq/heateqlapl.c'])
m2 = Extension('heateqrndr',
                    libraries=["cairo"],
                    sources = ['heateq/heateqrndr.c'])

setuptools.setup(name = 'Heateq',
      author='Julius Adorf',
      version = '0.1.dev',
      description = 'Heat equation simulator',
      include_dirs=[numpy.get_include(), '/usr/include/cairo'],
      entry_points = {
          'console_scripts': [
              'heateq = heateq.__main__:main',
          ]},
      ext_modules = [m1, m2],
      packages = setuptools.find_packages(),
      platforms = 'any',
      license='BSD License',
      keywords='heat equation')
