#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2019 R&D, NTC Inc. (ntc.ai)
#
# Author: qinluo <eric.x.sun@gmail.com>
#

import os
import platform
import re
import subprocess
import sys
from distutils.sysconfig import get_python_lib  # pylint: disable=import-error
from distutils.version import LooseVersion  # pylint: disable=no-name-in-module, import-error

import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):  # pylint: disable=too-few-public-methods
    def __init__(self, name, source_dir='', sources=None, **kwargs):
        if sources is None:
            sources = []
        Extension.__init__(self, name, sources=sources, **kwargs)
        self.source_dir = os.path.abspath(source_dir)


class CMakeBuild(build_ext):
    cuda_support = True
    support = '-DcuBERT_ENABLE_GPU=ON'
    support_py_pkg = '-DcuBERT_ENABLE_PYTHON_PKG=ON'
    py_site_package = get_python_lib()

    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                'CMake must be installed to build the following extensions: {0}'.format(
                    ', '.join(e.name for e in self.extensions)
                )
            )

        if platform.system() == 'Windows':
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError('CMake >= 3.1.0 is required on Windows')

        self.cuda_support = True
        try:
            subprocess.check_output(['nvcc', '--version'])
        except OSError:
            # 'no cuda supported, using MKL'
            self.cuda_support = False

        self.support = '-DcuBERT_ENABLE_GPU=ON' if self.cuda_support else '-DcuBERT_ENABLE_MKL_SUPPORT=ON'

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        ext_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + ext_dir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DPYTHON_SITE_PACKAGE=' + self.py_site_package
        ]  # yapf: disable
        cmake_args += [self.support, self.support_py_pkg]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == 'Windows':
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), ext_dir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO="{}"'.format(env.get('CXXFLAGS', ''), self.distribution.get_version())

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.source_dir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


SRC = 'src'
r_cc = re.compile(r'\.(cpp|c|cc|hpp)$')
src_files = [os.path.join(root, _file) for root, _, files in os.walk(SRC) for _file in files if files]
src_files = map(str, src_files)
src_cc = [x for x in src_files if r_cc.findall(x)]

ext_modules = [
    CMakeExtension(
        'cuBERT_pybind',
        '',
        ['python/cuBERT/pybind/cuBERT_pybind.cc'] + src_cc,
        include_dirs=[SRC],
        language='c++',
    ),
]

setuptools.setup(
    name='cuBERT',
    version='0.1',
    author='qinluo',
    author_email='eric.x.sun@gmail.com',
    description='python interface of cuBERT',
    ext_modules=ext_modules,
    cmdclass={'build_ext': CMakeBuild},
    packages=[str('cuBERT'), str('cuBERT.tests')],
    package_dir={str(''): str('python')},
    zip_safe=False
)
