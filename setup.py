from setuptools import setup, find_packages
import re
import os
import io


this_directory = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def get_version(package):
    """Return package version as listed in `__version__` in `init.py`."""
    init_py = open(os.path.join(package, '__init__.py')).read()
    return re.search("__version__ = ['\"]([^'\"]+)['\"]", init_py).group(1)


setup(
    name='tfdiffeq',
    version=get_version("tfdiffeq"),
    packages=find_packages(exclude=['tests']),
    url='https://github.com/titu1994/tfdiffeq',
    license='MIT',
    author='Somshubra Majumdar',
    author_email='titu1994@gmail.com',
    description='Tensorflow implementation of Partial Differential Equation Solvers with full GPU support',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['numpy>=1.16.2',
                      'scipy>=1.1.0',
                      'matplotlib>=3.0.0; python_version > "3.0"',
                      'six>=1.11.0'],
    extras_require={
        'uode': ['pysindy', 'tensorflow_probability'],
        'tests': ['six>=1.11.0'],
    },
    classifiers=(
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ),
    zip_safe=False,
    test_suite="tests",
)
