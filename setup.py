"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path

from setuptools import find_packages, setup

def parse_requirements(filename: str):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

reqs = parse_requirements("requirements.txt")

print(reqs)

# get version from version.py for package creation
version = {}
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = None
with open(os.path.join(base_dir, 'version.py')) as fp:
     exec(fp.read(), version)

# regular setup.py metadata
if __name__ == '__main__':
    tests_require = [
        'flake8',
        'mypy',
        'mypy-extensions',
        'mypy-protobuf',
        'types-PyYAML'
        'pylint',
        'pytest',
        'pytest-cov',
        'pytest-order',
        'yapf',
        'isort',
        'rope',
        'pre-commit',
        'pre-commit-hooks',
        'detect-secrets',
        'blacken-docs',
        'bashate',
        'fish',
        'watchdog',
        'speedscope',
        'pandas-profiling'
    ]

    docs_require = [
        'inari[mkdocs]',
        'mkdocs',
        'mkdocs-git-revision-date-localized-plugin',
        'mkdocs-macros-plugin',
        'mkdocs-material',
        'mkdocs-material-extensions',
        'mkdocs-mermaid-plugin',
        'mkdocs-pdf-export-plugin',
        'mktheapidocs[plugin]',
        'mkdocstrings',
        'mkdocs-autorefs',
        'mkdocs-coverage',
        'mkdocs-gen-files',
        'mkdocs-literate-nav',
        'mkdocs-section-index',
        'pymdown-extensions',
    ]

    setup(
        name="corl",
        author="ACT3",
        description="ACT3 RL Library Library",

        long_description=Path("README.md").read_text(),
        long_description_content_type="text/markdown",

        url="https://git.act3-ace.com/act3-rl/corl",

        license="Distribution C",

        setup_requires=[
            'setuptools_scm',
            'pytest-runner'
        ],

        version=version["__version__"],

        # add in package_data
        include_package_data=True,
        package_data={
            'corl': ['*.yml', '*.yaml']
        },

        packages=find_packages(),

        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],

        install_requires=reqs,
        extras_require={
            "testing":  tests_require,
            "docs":  docs_require,
        },
        python_requires='>=3.8',
    )

