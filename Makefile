all: install

.PHONY: install test

install:
	@echo "Installing Animate..."
	@python3 -m pip install -e .
	@echo "Done."

install_dev:
	@echo "Installing Animate for development..."
	@python3 -m pip install -e .[dev]
	@echo "Done."
	@echo "Setting up pre-commit..."
	@pre-commit install
	@echo "Done."

lint:
	@echo "Checking lint..."
	@ruff check
	@echo "PASS"

# `mpiexec -n N ... parallel[N]` only runs tests with @pytest.mark.parallel(nprocs=N)
test: lint
	@echo "Running all tests..."
	@python3 -m pytest -v --durations=20 -k "parallel[1] or not parallel" test
	@mpiexec -n 2 python3 -m pytest -v -m parallel[2] test
	@mpiexec -n 3 python3 -m pytest -v -m parallel[3] test
	@echo "Done."

coverage:
	@echo "Generating coverage report..."
	@python3 -m coverage erase
	@python3 -m coverage run --parallel-mode --source=animate -m pytest -v -k "parallel[1] or not parallel" test
	@mpiexec -n 2 python3 -m coverage run --parallel-mode --source=animate -m pytest -v -m parallel[2] test
	@mpiexec -n 3 python3 -m coverage run --parallel-mode --source=animate -m pytest -v -m parallel[3] test
	@python3 -m coverage combine
	@python3 -m coverage report -m
	@python3 -m coverage html
	@echo "Done."

tree:
	@tree -d .
