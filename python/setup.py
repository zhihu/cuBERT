import setuptools
import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np
import platform
import shutil

extra_compile_args = ['-std=c++11']
extra_link_args = []
libname = 'libcuBERT.so'
if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++']
    extra_link_args += ['-stdlib=libc++']
    libname = 'libcuBERT.dylib'

shutil.copyfile('../build/' + libname, './libcubert/' + libname)

setup(
    name='cuBERT',
    version='0.0.3',
    author='qinluo',
    author_email='eric.x.sun@gmail.com',
    description='python interface for cuBERT',
    packages=setuptools.find_packages(),
    package_data={'libcubert': ['libcuBERT.so', 'libcuBERT.dylib']},
    ext_modules = cythonize([Extension('cuBERT',
                                       sources=['cuBERT.pyx'],
                                       libraries=['cuBERT'],
                                       library_dirs=['../build'],
                                       include_dirs=[np.get_include()],
                                       language='c++',
                                       extra_compile_args=extra_compile_args,
                                       extra_link_args=extra_link_args)],
                            compiler_directives={'language_level' : sys.version_info[0]}),
)
