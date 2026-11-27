.PHONY: install lint format test health demo run docker-build clean

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

lint:
	python -m ruff check src tests main.py
	python -m ruff format --check src tests main.py
	python -m flake8 --max-line-length=120 src tests main.py

format:
	python -m ruff format src tests main.py
	python -m ruff check --fix src tests main.py

test:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=75

health:
	python main.py --health

demo:
	python main.py --demo --horizon both --threshold 0.30

run:
	python main.py --serve

docker-build:
	docker build -t latepay-detector:latest .

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in list(pathlib.Path('.').rglob('__pycache__')) + [pathlib.Path('.pytest_cache'), pathlib.Path('.ruff_cache'), pathlib.Path('.mypy_cache'), pathlib.Path('htmlcov')]]"
