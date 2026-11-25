.PHONY: setup data run test all
setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
data:
	.venv/bin/python -c 'from src.data_generator import save_data; save_data()'
run:
	.venv/bin/python -m src.run_pipeline
test:
	.venv/bin/pytest -q
all: setup data test run
