# Tests

This repo uses pytest. A default configuration is in `pytest.ini` and sets coverage to track `src`.

Quick start:

- Run the suite with coverage summary:
	- In this conda env: `conda run --name feedback-machinery pytest -q`
	- With explicit coverage report: `conda run --name feedback-machinery pytest -q --cov=src --cov-report=term-missing`

Notes:
- We include lightweight synthetic fixtures in `tests/conftest.py` to validate
	small helpers in `src/` (time slicing, kinematic derivatives, chop/merge, dPCA tensors).
- If you add new modules, prefer small, deterministic unit tests and extend fixtures as needed.