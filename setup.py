#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Record installation
import codecs
import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

with open(os.path.join(here, "record", "version.py")) as fp:
    exec(fp.read())

setup(
    name="record",
    version=__version__,  # noqa: F821
    author="Justin Chen",
    author_email="ch3njus@gmail.com",
    packages=find_packages(),
    package_data={"": ["LICENSE"],},
    url="https://github.com/ch3njust1n/record",
    license="MIT",
    entry_points={"console_scripts": ["record = record.cli:main",],},
    install_requires=["psutil", "pymongo", "torch"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: Microsoft",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python",
        "Topic :: Internet",
        "Topic :: Multimedia :: Video",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Terminals",
        "Topic :: Utilities",
    ],
    description=("Python 3 module for recording experiment results"),
    include_package_data=True,
    long_description_content_type="text/markdown",
    long_description=long_description,
    zip_safe=True,
    python_requires=">=3.6",
    project_urls={
        "Bug Reports": "https://github.com/ch3njust1n/record/issues"
    },
    keywords=["python3", "dictionary", "mongodb", "experiments", "reproducibility",],
)