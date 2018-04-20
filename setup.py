"""The packaging script for leabra7."""
from setuptools import setup, find_packages  # type: ignore

tests_require = [
    "hypothesis", "mypy", "pylint", "pytest", "pytest-cov",
    "pytest-mock", "tox", "yapf"
]

setup(
    name="Leabra7",
    version="0.1.dev1",
    packages=find_packages(exclude=["docs", "tests"]),
    install_requires=["numpy", "pandas", "scipy", "torch"],
    tests_require=tests_require,
    # We put the test requirements in the extra requirements also to access
    # them from outside tools, e.g. pip3 install .[dev]
    extras_require={
        "dev": tests_require
    })
