from ctypes import cdll
import os
import platform

libname = {
    'Linux': 'libcuBERT.so',
    'Darwin': 'libcuBERT.dylib',
}
cdll.LoadLibrary(os.path.join(os.path.abspath(os.path.dirname(__file__)), libname[platform.system()]))

from cuBERT import *
