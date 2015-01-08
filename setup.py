from setuptools import setup, find_packages

setup(
    name='FFTA',
    version='1.0',
    description='FF-trEFM Analysis Package',

    author='Durmus U. Karatay',
    author_email='ukaratay@uw.edu',
    license='MIT',

    packages=find_packages(exclude=['xop', 'docs', 'data']),

    install_requires=[
    'python=2.7.2',
    'numpy>=1.9.1',
    'scipy>=0.14.0',
    'igor>=0.2',
    'progressbar>=2.2'
    ],

    entry_points={
        'console_scripts': [
            'ffta-analyze=ffta:analyze:main',
        ],
    },
)