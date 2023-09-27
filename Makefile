all: install

.PHONY: install test

install:
	@echo "Installing dependencies..."
	@python3 -m pip install -r requirements.txt
	@echo "Done."
	@echo "Installing Animate..."
	@python3 -m pip install -e .
	@echo "Done."

lint:
	@echo "Checking lint..."
	@flake8 --ignore=E501,E226,E402,E741,F403,F405,W503
	@echo "PASS"

test: lint
	@echo "Running all tests..."
	@python3 -m pytest -v --durations=20 test
	@echo "Done."

coverage:
	@echo "Generating coverage report..."
	@python3 -m coverage erase
	@python3 -m coverage run --source=animate -m pytest -v test
	@python3 -m coverage html
	@echo "Done."

tree:
	@tree -d .
