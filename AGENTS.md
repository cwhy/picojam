# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

PicoJam is a JAX-based ML research monorepo for image experiments (MNIST, Fashion-MNIST, CIFAR-10). It contains Python modules (`jam/berries/`, `jam/berry_jam/`) and ~35 Jupyter notebooks (`jam/notebooks/`). Dependencies are managed via Poetry from `pico/pyproject.toml`.

### Running code

- All commands must run through **Poetry's virtualenv**: `cd /workspace/pico && poetry run <cmd>`
- Python scripts in `jam/berries/` and `jam/berry_jam/` use relative imports (e.g., `from pf import F`). When running them, add the `jam/berries` dir to `sys.path` or `PYTHONPATH`.
- JAX runs on **CPU backend** in this environment (no GPU). Scripts still work, just slower for large experiments.
- The `my_datasets.py` module uses `cache_dir="$HOME/.cache/huggingface/datasets"` (a literal `$HOME` string, not shell-expanded). A symlink `$HOME -> /home/ubuntu` must exist at the workspace root (`/workspace/$HOME`) for dataset caching to work. Create it with: `ln -sf /home/ubuntu /workspace/\$HOME`

### Linting

- `cd /workspace/pico && poetry run mypy ../jam/berries/ --ignore-missing-imports` — pre-existing type errors in `plot_utils.py`, `my_datasets.py`, `init_utils.py` are expected.

### Jupyter notebooks

- `cd /workspace/pico && poetry run jupyter notebook --no-browser --port=8888 --ip=0.0.0.0 --ServerApp.token=''`
- Notebooks are in `jam/notebooks/` and `jam/berry_taste/`.

### Key gotchas

- `pyqt5` is installed but has no display server in Cloud VMs. Use `MPLBACKEND=Agg` if matplotlib scripts hang trying to open a GUI window.
- WandB integration in training scripts (`embed.py`, etc.) defaults to `use_wandb = True`. Set it to `False` or mock `wandb` for local testing without a WandB API key.
- The `pf.py` module's `F` class and `_` placeholder are used extensively throughout. `F` wraps functions for point-free composition, vmap, and partial application.
