import setuptools
from distutils.core import setup

setup(
    name='solartf',
    version='0.0.1',
    packages=setuptools.find_packages(),
    install_requires=[
        'tensorflow==2.6.0'
    ]
)
