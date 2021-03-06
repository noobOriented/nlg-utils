.DEFAULT_GOAL := all

.PHONY: lint
lint:
	flake8

.PHONY: test
test:
	pytest --cov=. --cov-fail-under=20

.PHONY: test-report
test-report:
	pytest -W ignore --cov=. --cov-report term-missing

.PHONY: clean
clean:
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]' `
	rm -f `find . -type f -name '*~' `
	rm -f `find . -type f -name '.*~' `
	rm -rf .cache
	rm -rf htmlcov
	rm -rf *.egg-info
	rm -f .coverage
	rm -f .coverage.*
	rm -rf build
	rm -rf dist
	python setup.py clean
