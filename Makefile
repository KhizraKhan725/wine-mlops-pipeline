.PHONY: install lint test train clean

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

lint:
	flake8 src/ tests/ --max-line-length=100

test:
	pytest -v

train:
	python src/train.py

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]; shutil.rmtree('.pytest_cache', ignore_errors=True)"
