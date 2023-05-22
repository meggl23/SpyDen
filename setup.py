from setuptools import setup, find_packages

setup(
    name='spyne',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'networkx',
        'scikit-image==0.19.2',
        'regex',
        'torch==1.13.0',
        'torchvision==0.14.0',
        'tifffile',
        'opencv',
        'pyqt',
        'pyqt5-sip'
    ],
    entry_points={
        'console_scripts': [
            'spyne = to_pyqt:main'
        ]
    },
    author='Maximilian Eggl',
    author_email='maximilian.eggl@uni-mainz.de',
    description='Analyse spines and dendrites',
    license='MIT',
)
