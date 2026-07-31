from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='algos',
    ext_modules=cythonize('algos/algos.pyx', compiler_directives={'language_level': 3}),
    include_dirs=[np.get_include()],
)
