from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

import numpy

fitting = Extension('ffta.utils.fitting_c', ['ffta/utils/fitting_c.pyx'],
                    include_dirs=[numpy.get_include()])
setup(
    name='FFTA',
    version='1.2',
    description='FF-trEFM Analysis Package',

    author='Durmus U. Karatay',
    author_email='ukaratay@uw.edu',
    license='MIT',

    packages=find_packages(exclude=['xop', 'docs', 'data']),

    install_requires=['numpy>=1.9.2',
                      'scipy>=0.15.1',
                      'igor>=0.2',
                      'progressbar>=2.2',
                      'cython>=0.23'],

    ext_modules=cythonize([fitting]),

    entry_points={
        'console_scripts': [
            'ffta-analyze = ffta.analyze:main',
        ],
    },
)
