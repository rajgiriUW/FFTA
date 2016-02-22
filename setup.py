from setuptools import setup, Extension, find_packages

import numpy

fitting = Extension('ffta.utils.fitting_c', ['ffta/utils/fitting_c.pyx'],
                    include_dirs=[numpy.get_include()])
setup(
    name='FFTA',
    version='1.3',
    description='FF-trEFM Analysis Package',

    author='Rajiv Giridharagopal',
    author_email='rgiri@uw.edu',
    license='MIT',

    packages=find_packages(exclude=['xop', 'docs', 'data']),

    install_requires=['numpy>=1.9.2',
                      'scipy>=0.15.1',
                      'igor>=0.2',
                      'progressbar>=2.2'],

    entry_points={
        'console_scripts': [
            'ffta-analyze = ffta.analyze:main',
        ],
    },
)
