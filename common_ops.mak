.PHONY: install setup test docker clean

install:
	python install.py

setup:
	python main.py setup

test:
	python main.py test

docker:
	python main.py create-docker
	docker-compose build
	docker-compose up -d

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	rm -rf build dist .coverage htmlcov

format:
	black .
	isort .

lint:
	black . --check
	isort . --check
	mypy .

jupyter:
	jupyter lab

gui:
	streamlit run src/gui.py