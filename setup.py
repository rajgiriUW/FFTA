from setuptools import setup, find_packages

from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(this_directory, 'ffta/__version__.py')) as f:
    __version__ = f.read().split("'")[1]

setup(
    name='FFTA',
    version=__version__,
    description='Fast Free Transient Analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',

    author='Rajiv Giridharagopal',
    author_email='rgiri@uw.edu',
    license='MIT',
    url='https://github.com/rajgiriUW/ffta/',

    packages=find_packages(exclude=['xop', 'docs', 'data', 'notebooks']),

    install_requires=['numpy>=1.20.0',
                      'scipy>=1.4.1',
                      'igor2>=0.5.2',
                      'numexpr>=2.7.1',
                      'pyUSID>=0.0.10',
                      'pywavelets>=1.1.1',
                      'sidpy>=0.0.5',
                      'pandas>=1.1.0',
                      'BGlib>=0.0.3',
                      'torch>=1.6.0',
                      'scikit-image>=0.19.2'
                      ],
    test_suite='pytest',
    # entry_points={
    #      'console_scripts': [
    #          'ffta-analyze = ffta.analyze:main',
    #      ],
    # },

)
