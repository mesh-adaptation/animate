all: run

run:
	@echo "Running all demos..."
	@for file in ./*.py; do python3 $$file; done
	@echo "Done."

clean:
	@echo "Running demo caches..."
	@rm -rf __pycache__
	@echo "Done."
	@echo "Removing demo outputs..."
	@rm -rf *.jpg
	@rm -rf outputs*
	@echo "Done."
