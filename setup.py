from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

import numpy

setup_options = dict(
    name="stat/viterbi",
    ext_modules=cythonize(Extension("viterbi", ["stat/viterbi.pyx"])),
    include_dirs=[numpy.get_include()],
    script_args=["build_ext"],
    options={"build_ext": {"build_lib": "stat"}}
)

setup(**setup_options)
