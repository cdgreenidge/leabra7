PROJECT=leabra7
PYTHON=python3

all: format check test

check: check-leabra check-notebook

check-leabra:
	@mypy $(PROJECT)
	@pylint $(PROJECT)
	@yapf --parallel --recursive --diff $(PROJECT) tests

check-notebook:
	@mypy notebooks/*.py
	@pylint notebooks/*.py
	@yapf --parallel --recursive --diff notebooks/*.py

clean:
	rm -rf .cache .mypy_cache *.egg-info .tox $(PROJECT)/__pycache__ \
		tests/__pycache__ $(PROJECT)/*.pyc tests/*.pyc

notebooks/%.py : notebooks/%.ipynb
	python scripts/notebook_to_python.py < $< > $@

notebooks/%.ipynb : notebooks/%.py
	python scripts/python_to_notebook.py < $< > $@

distclean: clean
	rm -rf $(VIRTUALENV_DIR)/$(PROJECT)

format: format-leabra format-notebook

format-leabra:
	@yapf --parallel --recursive --in-place $(PROJECT) tests

format-notebook:
	@yapf --parallel --recursive --in-place notebooks/*.py

test:
	@pytest --cov-report term-missing --cov=$(PROJECT) tests
