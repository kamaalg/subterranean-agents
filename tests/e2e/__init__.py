"""End-to-end reproduction tests (release gate).

Tagged ``@pytest.mark.e2e``. These run a full compile→generate→train→eval
pipeline and compare measured criterion means against the paper's published
numbers, failing on a >5% regression. They require a GPU-trained model and are
skipped by default; see :mod:`tests.e2e.test_reproductions`. The regression math
lives in :mod:`tests.e2e.regression` and is unit-tested in
``tests/unit/test_regression_gate.py``.
"""
