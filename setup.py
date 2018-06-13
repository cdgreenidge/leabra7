"""The packaging script for leabra7."""
from setuptools import setup, find_packages  # type: ignore

setup(
    name="Leabra7",
    version="0.1.dev1",
    packages=find_packages(exclude=["docs", "tests"]),
    install_requires=[
        "numpy>=1.14", "pandas>=0.23", "scipy>=1.1", "pytorch>=0.4"
    ])
