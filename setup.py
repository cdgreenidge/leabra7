from setuptools import setup, find_packages

setup(
    name="Leabra7",
    version="0.1.dev1",
    packages=find_packages(exclude=["docs", "tests"]),
    install_requires=["numpy", "scipy"],
    # We put the test requirements in the extra requirements also to access
    # them from outside tools, e.g. pip3 install .[dev]
    extras_require={
        "dev": ["mypy", "pylint", "pytest", "pytest-mock", "tox", "yapf"]
    })
