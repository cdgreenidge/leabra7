PROJECT=leabra7
PYTHON=python3
VIRTUALENV_DIR=$(HOME)/.virtualenvs

all: check format test

check:
	@mypy $(PROJECT)
	@pylint $(PROJECT)
	@yapf --parallel --recursive --diff $(PROJECT) tests

clean:
	rm -rf .cache .mypy_cache *.egg-info .tox $(PROJECT)/__pycache__ \
		tests/__pycache__ $(PROJECT)/*.pyc tests/*.pyc

distclean: clean
	rm -rf $(VIRTUALENV_DIR)/$(PROJECT)

format:
	@yapf --parallel --recursive --in-place $(PROJECT) tests

test:
	@pytest --cov=$(PROJECT)
