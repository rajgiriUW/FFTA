from setuptools import setup, find_packages

import numpy

setup(
    name='FFTA',
    version='1.5',
    description='FF-trEFM Analysis Package',

    author='Rajiv Giridharagopal',
    author_email='rgiri@uw.edu',
    license='MIT',

    packages=find_packages(exclude=['xop', 'docs', 'data']),

    install_requires=['numpy>=1.9.2',
                      'scipy>=0.15.1',
                      'igor>=0.2',
                      'pywavelets>=0.3.0',
                      'numexpr',
		      'watchdog'],

    entry_points={
        'console_scripts': [
            'ffta-analyze = ffta.analyze:main',
        ],
    },
)
