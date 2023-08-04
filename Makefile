all: install

install:
	@echo "Installing dependencies..."
	@python3 -m pip install -r requirements.txt
	@echo "Done."
	@echo "Installing Animate..."
	@python3 -m pip install -e .
	@echo "Done."
