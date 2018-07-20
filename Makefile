PROJECT=leabra7
PYTHON=python3

all: format check test

check: check-leabra check-notebook check-script

check-leabra:
	@mypy $(PROJECT)
	@pylint $(PROJECT)
	@yapf --parallel --recursive --diff $(PROJECT) tests

check-notebook:
	@mypy notebooks/*.py
	@pylint notebooks/*.py
	@yapf --parallel --recursive --diff notebooks/*.py

check-script:
	@mypy scripts/*.py
	@pylint scripts/*.py
	@yapf --parallel --recursive --diff scripts/*.py

clean:
	rm -rf .cache .mypy_cache *.egg-info .tox $(PROJECT)/__pycache__ \
		tests/__pycache__ $(PROJECT)/*.pyc tests/*.pyc

convert-notebook:
	@python scripts/notebook_to_python.py

distclean: clean
	rm -rf $(VIRTUALENV_DIR)/$(PROJECT)

format: format-leabra format-notebook format-script

format-leabra:
	@yapf --parallel --recursive --in-place $(PROJECT) tests

format-notebook:
	@yapf --parallel --recursive --in-place notebooks/*.py

format-script:
	@yapf --parallel --recursive --in-place scripts/*.py

test:
	@pytest --cov-report term-missing --cov=$(PROJECT) tests

reverse-notebook:
	@python scripts/python_to_notebook.py
