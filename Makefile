# Reproduction entry points. See README.md for what each target produces.
PY := .venv/bin/python

.PHONY: help venv test validate validate-span experiments span-experiments figures span-figures summary reproduce smoke-test clean

help:
	@echo "make venv        create the pinned virtual environment"
	@echo "make test        run the test suite (invariants, closed forms, cross-fidelity)"
	@echo "make validate    reproduce the external validation against published measurements"
	@echo "make validate-span  the span-tier registry and the list of what it declines"
	@echo "make span-experiments  the latency-regime atlas (results/raw/s1..s7)"
	@echo "make span-figures      fig8 and fig9 from the span results"
	@echo "make smoke-test  fast subset: tests, validation, two experiments, figures (<1 min)"
	@echo "make reproduce   full program: tests, validation, all experiments, figures, paper"
	@echo "make paper       build the PDF"
	@echo "make clean       remove generated results and figures"

venv:
	python3.12 -m venv .venv
	.venv/bin/pip install --quiet --upgrade pip
	.venv/bin/pip install --quiet -e ".[dev]"

test:
	PYTHONPATH=src $(PY) -m pytest tests/ -q

validate:
	$(PY) validation/validate_llama3.py

validate-span:
	$(PY) validation/validate_span.py

experiments:
	$(PY) experiments/run_experiments.py

span-experiments:
	$(PY) experiments/run_span_experiments.py

summary:
	$(PY) experiments/summarize.py > /dev/null && echo "wrote results/processed/"

figures:
	$(PY) figures/palette_check.py > /dev/null
	$(PY) figures/make_figures.py

span-figures:
	$(PY) figures/palette_check.py > /dev/null
	$(PY) figures/make_span_figures.py

paper:
	cd paper && tectonic -X compile main.tex --outdir .

smoke-test: test validate validate-span
	$(PY) experiments/run_experiments.py --smoke
	$(PY) experiments/run_span_experiments.py --smoke
	@echo "smoke test complete"

reproduce: validate validate-span experiments span-experiments summary figures span-figures test paper
	@echo "full reproduction complete"

clean:
	rm -rf results/raw/* results/processed/* figures/*.pdf figures/*.png paper/main.pdf
