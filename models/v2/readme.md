Build Instructions:

    python -m venv .venv
	source .venv/bin/activate
	python -m pip install -U pip
	pip install -r requirements.txt
	pip install -e .
	python -c "import scada_tcn; print(scada_tcn.__version__)"
