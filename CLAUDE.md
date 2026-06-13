# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`imcp` is a small, dependency-light PyPI package (`numpy` + `matplotlib` only) implementing the **(I)MCP curve** — a generalization of the ROC curve to multiclass and class-imbalanced classification, from Aguilar-Ruiz & Michalak (Scientific Reports 14:10759, 2024; IEEE Access 10:68915, 2022). The whole implementation is one module: `imcp/imcp.py`.

## Commands

Tests use the `unittest` framework and read their CSV fixtures via **relative paths**, so they must be run from inside `tests/`:

```bash
cd tests
python -m unittest unittests.py                          # run all tests
python -m unittest unittests.TestMCPCurve.test_mcp_area   # run a single test
```

Install / build / docs:

```bash
pip install -e .              # editable dev install (deps: numpy, matplotlib; tests also need pandas)
python -m build               # build sdist+wheel (publish is automated by GitHub Actions on Release)
cd docs-sphinx && make html   # rebuild docs; committed HTML lives in docs/ and is served via GitHub Pages
```

Note: this machine forbids polluting system/Homebrew/MacPorts Python — run everything through `uv` (e.g. `uv run --no-project --with numpy --with matplotlib --with pandas python <script>`) or a project-local `.venv`. See the global rule in `~/.claude/CLAUDE.md`.

## Architecture

**Public API** (defined in `imcp/imcp.py`, re-exported by `imcp/__init__.py`): `mcp_curve`, `imcp_curve`, `mcp_score`, `imcp_score`, `plot_mcp_curve`, `plot_imcp_curve`, `plot_curve`.

**The computation pipeline** (shared by MCP and IMCP):
1. `_map_class_labels` validates inputs and integer-encodes labels; `y_true` is then one-hot encoded.
2. `_get_y_values` computes the y-axis as `1 - HellingerDistance(one_hot(y_true), sqrt(y_score)) / sqrt(2)` per sample, then sorts samples by `np.lexsort((y_true, curve_y))`. This y-axis is identical for both curves.
3. The **only difference between MCP and IMCP is the x-axis**:
   - **MCP** spaces samples uniformly: `arange(n) / (n-1)`.
   - **IMCP** gives each sample a width `1/(n_classes * class_count)` (`_get_class_widths`) and uses the cumulative center of those widths. Minority-class samples therefore occupy proportionally *more* horizontal space — this is the imbalance correction. IMCP also pins the curve to the points `(0, φ₁)` and `(1, φₙ)`, so its arrays are 2 elements longer than MCP's.
4. `*_score` functions are the area under the curve via `np.trapz(curve_y, x=curve_x)`.

**Input conventions** (enforced at runtime):
- `y_true`: shape `(n_samples,)`. `y_score`: shape `(n_samples, n_classes)`, **each row must sum to 1.0** within `abs_tolerance` (default `1e-8`; pass a larger value like `1e-6` for noisy classifier probabilities).
- `y_score` columns must be ordered by the numeric/lexicographic order of the class labels.
- If a class appears in `y_score` columns but never in `y_true`, you must pass `labels=[...]` enumerating all columns' labels (same dtype as `y_true`).
- `y_true`/`y_score` may be pandas `Series`/`DataFrame` or numpy arrays — both flow through unchanged (the tests pass pandas objects).

**Plotting**: `plot_mcp_curve` / `plot_imcp_curve` accept either a single `(n, k)` score array (one classifier) **or a `dict {algo_name: score_array}`** to overlay several classifiers on one figure; the area is auto-computed and written into each legend entry. Both delegate to `plot_curve`, which dispatches on 1D-vs-2D input via `_get_dimensions` and saves to `output_fig_path` if given (else `plt.show()`).

## Data fixtures (`tests/`)

CSVs follow a `y_true` + `y_score_1..k` column convention. `test_results.csv` has 7 classes; `test_imbalanced_class_probs.csv` has 3 highly imbalanced classes; `test_mcp_curve.csv` is a precomputed ground-truth curve; `A/B/C/D_exp-*.base-100.csv` are additional experiment inputs.
