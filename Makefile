PROJECT=leabra7
PYTHON=python3

all: all-leabra all-notebook

all-leabra: check-leabra format-leabra test-leabra

all-notebook: convert-notebook finish-notebook

check-leabra:
	@mypy $(PROJECT)
	@pylint $(PROJECT)
	@yapf --parallel --recursive --diff $(PROJECT) tests

check-notebook:
	@mypy notebooks/*.py
	@pylint notebooks/*.py

clean:
	rm -rf .cache .mypy_cache *.egg-info .tox $(PROJECT)/__pycache__ \
		tests/__pycache__ $(PROJECT)/*.pyc tests/*.pyc

convert-notebook:
	@python scripts/notebook_to_python.py

distclean: clean
	rm -rf $(VIRTUALENV_DIR)/$(PROJECT)

finish-notebook: check-notebook format-notebook reverse-notebook

format-leabra:
	@yapf --parallel --recursive --in-place $(PROJECT) tests

format-notebook:
	@yapf --parallel --recursive --in-place notebooks/*.py

test-leabra:
	@pytest --cov-report term-missing --cov=$(PROJECT) tests

reverse-notebook:
	@python scripts/python_to_notebook.py
