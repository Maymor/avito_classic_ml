"""Выполнить notebook в текущем venv и сохранить outputs без глобального Jupyter."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Эти каталоги задаём до запуска ядра. Jupyter/IPython не будут пытаться писать
# настройки, runtime-файлы или кеш matplotlib в домашний каталог пользователя.
for key, relative in {
    "JUPYTER_CONFIG_DIR": ".cache/jupyter/config",
    "JUPYTER_DATA_DIR": ".cache/jupyter/data",
    "JUPYTER_RUNTIME_DIR": ".cache/jupyter/runtime",
    "IPYTHONDIR": ".cache/ipython",
    "MPLCONFIGDIR": ".cache/matplotlib",
}.items():
    directory = Path(sys.prefix) / relative
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(directory)

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


path = ROOT / "solution.ipynb"
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)
manager = KernelManager(kernel_name="python3")
# Не полагаемся на то, какой `python` первым стоит в PATH у вызывающей оболочки.
manager.kernel_spec.argv[0] = sys.executable


def report_progress(cell, cell_index, **kwargs):
    if cell.cell_type == "code":
        print(f"Execute code cell {cell_index + 1}/{len(notebook.cells)}", flush=True)


client = NotebookClient(notebook, km=manager, timeout=1800, allow_errors=False, coalesce_streams=True,
                        startup_timeout=45, resources={"metadata": {"path": str(ROOT)}},
                        on_cell_start=report_progress)
print("Start a local Jupyter kernel in the active venv", flush=True)
try:
    client.execute()
finally:
    # Даже при ошибке сохраняем настоящее состояние, а не фиктивные outputs.
    nbformat.write(notebook, path)
    # KernelManager передан явно, поэтому NotebookClient не владеет ядром.
    # Закрываем его сами, чтобы после исполнения не оставался фоновый процесс.
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)

nbformat.validate(notebook)
print("Notebook executed top-to-bottom and saved", flush=True)
