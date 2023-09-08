"""
  The setup script to install for python
"""

from __future__ import absolute_import

from pathlib import Path

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

SDK_VERSION = '1.0'
DESC = Path('./README.md').read_text(encoding='utf-8')

setup(
    name='circulant-solver',
    version=SDK_VERSION,
    install_requires=[
        'numpy',
        'qiskit',
        'qiskit-aer',
        'cvxopt'
    ],
    python_requires='>=3.8, <3.11',
    packages=find_packages(),
    license='Apache License 2.0',
    author='Xiufan Li, Po-Wei Huang',
    author_email='e1117166@u.nus.edu',
    description='circulant-solver is a Python-based quantum software development kit (SDK). '
                'It provides a full-stack programming experience for numerical simulation of the'
                'algorithm for solving k-banded circulant linear systems of equations via high-performance simulators, '
                "and various hardware platforms with quantum computers.",
    long_description=DESC,
    long_description_content_type='text/markdown'
)
