# mpl-tripartite

Custom matplotlib projection for tripartite (four-coordinate) log-log plots with diagonal gridlines for constant acceleration and constant displacement. Gridlines are rendered natively as part of the axes drawing pipeline — they auto-update on zoom, pan, and resize.

## Installation

```bash
pip install mpl-tripartite
```

## Development

This project uses [uv](https://docs.astral.sh/uv/) as its dev/build runner and [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# install dev environment
uv sync

# run tests
uv run pytest

# lint + format
uv run ruff check .
uv run ruff format .

# type-check
uv run mypy src

# build sdist + wheel
uv build
```

## Usage

### Pseudo-velocity SRS

```python
import mpl_tripartite  # registers the 'tripartite' projection
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(subplot_kw=dict(projection='tripartite'))

freq = np.logspace(1, 4, 200)
pv = 20 / (2 * np.pi * freq)  # constant acceleration curve
ax.loglog(freq, pv, label='100 g')

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Pseudo-Velocity (in/s)')
ax.set_xlim(10, 10000)
ax.set_ylim(0.1, 100)
ax.legend()
plt.show()
```

### Earthquake design spectrum

```python
from mpl_tripartite import TripartiteProjection

fig, ax = plt.subplots(subplot_kw=dict(
    projection=TripartiteProjection(
        v_unit='m/s',
        xlabel='Frequency (Hz)',
        ylabel='Sv (m/s)',
        neg_diag_label='m/s²',
        pos_diag_label='m',
        g_normalize=False,
    )
))
```

## License

MIT
