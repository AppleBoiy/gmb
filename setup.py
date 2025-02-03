from setuptools import setup, find_packages

setup(
    name='graph-matching-benchmark',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'networkx',
        'numpy',
        'scikit-learn',
        'matplotlib',
    ],
    description='A benchmark for graph matching algorithms.',
    author='Your Name',
    license='MIT',
)