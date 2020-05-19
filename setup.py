from setuptools import setup, find_packages

setup(
    name='FFTA',
    version='0.1',
    description='Fast Free Transient Analysis',

    author='Rajiv Giridharagopal',
    author_email='rgiri@uw.edu',
    license='MIT',

    packages=find_packages(exclude=['xop', 'docs', 'data', 'notebooks']),

    install_requires=['numpy>=1.18.1',
                      'scipy>=1.4.1',
                      'igor>=0.3',
                      'numexpr>=2.7.1',
                      'watchdog>=0.10.2',
                      'pyUSID>=0.0.8',
                      'pycroscopy>=0.60',
                      'pywavelets>=1.1.1'],

    entry_points={
        'console_scripts': [
            'ffta-analyze = ffta.analyze:main',
        ],
    },

)
