import numpy
import os
from setuptools import setup, find_packages, Extension

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as reqh:
    install_requires = reqh.readlines()

libsptlzsrc = os.path.join('src', 'c++', 'libspatialize.cpp')
macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]

libspatialize_extensions = [
    Extension(name='libspatialize',
              sources=[libsptlzsrc],
              include_dirs=[ os.path.join('.', 'include'), numpy.get_include()],
              extra_compile_args=['-std=c++14'],
              define_macros=macros,
              ),
]

if __name__ == '__main__':
    setup(
        name='spatialize',
        version='1.0.2',
        author='ALGES Laboratory',
        author_email='dev@alges.cl',
        description='Python wrapper for ESI',
        keywords="ESI ensemble spatial interpolation",
        url="http://www.alges.cl/",
        long_description=open(os.path.join(os.path.dirname(os.path.realpath(
            __file__)), "README.md")).read(),
        ext_modules=libspatialize_extensions,
        package_dir={'spatialize': os.path.join('src', 'python', 'spatialize')},
        packages=find_packages(os.path.join('src', 'python'), exclude=[".DS_Store", "__pycache__"]),
        include_package_data=True,
        scripts=[],
        install_requires=install_requires,
    )
