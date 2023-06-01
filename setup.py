# from distutils.core import setup, Extension
import numpy
import os
from setuptools import setup, find_packages
from distutils.extension import Extension

libsrc = os.path.join('src', 'c++', 'libspatialize.cpp')
macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]

libspatialize_extensions = [
    Extension(name='libspatialize',
              sources=[libsrc],
              include_dirs=[numpy.get_include(), os.path.join('.', 'include')],
              extra_compile_args=['-std=c++14'],
              define_macros=macros,
              ),
]

if __name__ == '__main__':
    setup(
        name='specialize',
        version='0.1',
        author='ALGES Laboratory',
        author_email='dev@alges.cl',
        description='Python wrapper for ESI',
        keywords="ESI ensemble spatial interpolation",
        url="http://www.alges.cl/",
        long_description=open(os.path.join(os.path.dirname(os.path.realpath(
            __file__)), "README.txt")).read(),
        ext_modules=libspatialize_extensions,
        # packages=['spatialize'],
        # package_dir={'justesi': 'src'},
        # packages=find_packages('..', exclude=[".DS_Store", "__pycache__"]),
        include_package_data=True,
        scripts=[],
        install_requires=[
            'numpy >= 1.8.0',
        ],
    )
