from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np

# LDFLAGS="-L../build/" python setup.py build_ext -i
# LD_LIBRARY_PATH=../build/ python cuBERT_test.py

setup(
    ext_modules = cythonize([Extension("cuBERT", 
                                       sources=["cuBERT.pyx"], 
                                       libraries=["cuBERT"],
                                       include_dirs=[np.get_include()],
                                       language="c++",
                                       extra_compile_args=["-std=c++11", "-stdlib=libc++"],
                                       extra_link_args=["-stdlib=libc++"])])
)
