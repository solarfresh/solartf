import setuptools
from distutils.core import setup

setup(
    name='solartf',
    version='0.0.3',
    packages=setuptools.find_packages(),
    install_requires=[
        'opencv-python==4.5.3.56',
        'pandas==1.3.2',
        'matplotlib==3.4.3',
        'numpy==1.19.5',
        'scikit-learn==0.24.2',
        'scipy==1.7.1',
        'tensorflow==2.6.0'
    ]
)
