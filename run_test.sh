pylint leabra7
mypy leabra7
yapf --parallel --recursive --diff leabra7 tests
pytest --cov=${SP_DIR}/leabra7 --cov-report=
if [ "$TRAVIS" = true ]; then codecov; fi
