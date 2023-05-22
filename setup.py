from setuptools import setup, find_packages

setup(
    name='spyne',
    version='0.0.2',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'networkx',
        'scikit-image==0.19.2',
        'regex',
        'torch==1.13.1',
        'torchvision==0.14.1',
        'tifffile',
        'opencv',
        'pyqt',
        'pyqt5-sip'
    ]
    author='Maximilian Eggl',
    author_email='maximilian.eggl@uni-mainz.de',
    description='Analyse spines and dendrites',
    license='MIT',
)
