all: install

.PHONY: install test

install:
	@echo "Installing Animate..."
	@python3 -m pip install -e .
	@echo "Done."
	@echo "Setting up pre-commit..."
	@pre-commit install
	@echo "Done."

lint:
	@echo "Checking lint..."
	@ruff check
	@echo "PASS"

convert_demos:
	@echo "Converting demos into integration tests..."
	@mkdir -p test/demos
	@cd demos && for file in *.py; do \
		cp $$file ../test/demos/test_demo_$$file; \
	done
	@cd test && for file in demos/*.py; do \
		bash to_test.sh $$file; \
		ruff --fix $$file; \
	done
	@echo "Done."

test: lint convert_demos
	@echo "Running all tests..."
	@python3 -m pytest -v -n auto --durations=20 test
	@cd test && make clean
	@make clean
	@echo "Done."

coverage: convert_demos
	@echo "Generating coverage report..."
	@python3 -m coverage erase
	@python3 -m coverage run --source=animate -m pytest -v test
	@python3 -m coverage html
	@cd test && make clean
	@make clean
	@echo "Done."

tree:
	@tree -d .

clean:
	@rm -f *.jpg *.pvd *.vtu
	@rm -rf test/demos
