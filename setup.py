import Cython.Build
from setuptools import setup
from setuptools.extension import Extension

import numpy as np

long_description = "This is a long description"

setup(
    name='omniCLIP_test',
    version="0.1a",
    description='Test of the omniCLIP project',
    long_description=long_description,
    author='Philipp Boss',
    author_email='joel.simoneau@mdc-berlin.de',
    url='https://github.com/simojoe/omniCLIP',
    cmdclass={'build_ext': Cython.Build.build_ext},
    package_dir={'omniCLIP': 'omniCLIP'},
    packages=['omniCLIP', 'omniCLIP.data_parsing', 'omniCLIP.omni_stat'],
    ext_modules=[Extension(
        'omniCLIP.viterbi',
        sources=['omniCLIP/omni_stat/viterbi.pyx'],
        include_dirs=[np.get_include()],
    )],
    zip_safe=False,
    entry_points ={
        'console_scripts': [
            'omniCLIP = omniCLIP.omniCLIP:main'
        ]
    },
    setup_requires=['numpy>=1.18', 'cython>=0.24.1'],
    install_requires=[
        'numpy>=1.18', 'nose>=0.11', 'cython>=0.24.1',
        'statsmodels>=0.11.0', 'scipy>=1.4.1', 'scikit-learn>=0.22.1',
        'pysam>=0.15.3', 'pandas>=1.0.2', 'intervaltree>=3.0.2', 'h5py>=2.10.0', 'gffutils>=0.10.1'],
)
