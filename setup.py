import Cython.Build
from setuptools import setup
from setuptools.extension import Extension

import numpy as np

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

version = {}
with open("version.py") as fp:
    exec(fp.read(), version)

setup(
    name='omniCLIP',
    version=version['__version__'],
    description='omniCLIP is a CLIP-seq Bayesian peak caller.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Philipp Boss',
    author_email='philipp.drewe@googlemail.com',
    url='https://github.com/philippdre/omniCLIP',
    cmdclass={'build_ext': Cython.Build.build_ext},
    package_dir={'omniCLIP': 'omniCLIP'},
    packages=['omniCLIP', 'omniCLIP.data_parsing', 'omniCLIP.omni_stat'],
    ext_modules=[Extension(
        'omniCLIP.viterbi',
        sources=['omniCLIP/omni_stat/viterbi.pyx'],
        include_dirs=[np.get_include()],
    )],
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'omniCLIP = omniCLIP.omniCLIP:main'
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: GNU General Public License (GPL)",
    ],
    setup_requires=['numpy>=1.18', 'cython>=0.24.1'],
    install_requires=[
        'numpy>=1.18', 'nose>=0.11', 'cython>=0.24.1', 'h5py>=2.10.0',
        'statsmodels>=0.11.0', 'scipy>=1.4.1', 'scikit-learn>=0.22.1',
        'pysam>=0.15.3', 'pandas>=1.0.2', 'intervaltree>=3.0.2',
        'gffutils>=0.10.1', 'biopython>=1.76'],
)
