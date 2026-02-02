PYTHON ?= python3

bench-small:
	$(PYTHON) examples/bench/bench_dml_vs_rag.py --corpus-size 50 --queries 5 --output examples/bench/results-small.csv

bench-large:
	$(PYTHON) examples/bench/bench_dml_vs_rag.py --corpus-size 250 --queries 25 --output examples/bench/results-large.csv
