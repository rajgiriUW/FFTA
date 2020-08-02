from setuptools import setup, find_packages

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='FFTA',
    version='0.3.0',
    description='Fast Free Transient Analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',

    author='Rajiv Giridharagopal',
    author_email='rgiri@uw.edu',
    license='MIT',
	url='https://github.com/rajgiriUW/ffta/',

    packages=find_packages(exclude=['xop', 'docs', 'data', 'notebooks']),

    install_requires=['numpy>=1.18.1',
                      'scipy>=1.4.1',
                      'igor>=0.3',
                      'numexpr>=2.7.1',
                      'watchdog>=0.10.2',
                      'pyUSID>=0.0.8',
                      'pycroscopy>=0.60',
                      'pywavelets>=1.1.1'],

    # entry_points={
    #      'console_scripts': [
    #          'ffta-analyze = ffta.analyze:main',
    #      ],
    # },

)
