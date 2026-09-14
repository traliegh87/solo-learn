# Fix: `pip install -e .` fails with `ModuleNotFoundError: No module named 'pkg_resources'`

## Symptom

```
pip3 install -e .[umap,h5]
...
  File "<string>", line 21, in <module>
ModuleNotFoundError: No module named 'pkg_resources'
```

Happens during `Getting requirements to build editable`, regardless of Python version (reproduced on both Python 3.14 and a freshly created Python 3.10 env).

## Root cause

`setup.py:21` does `from pkg_resources import parse_requirements` and uses it at line 28 to parse `requirements.txt`. There's no `pyproject.toml` pinning a `[build-system] requires` version, so pip's build-isolation step downloads whatever `setuptools` is newest on PyPI into a throwaway venv to run `setup.py` — and current `setuptools` releases no longer ship `pkg_resources`. This breaks on **any** environment until that import is removed, not just newer Python versions.

(Separately, if you're on an unsupported Python version like 3.14: `setup.py` classifiers only declare 3.8–3.11, and `requirements.txt` pins `torch>=1.10.0`, `lightning==2.1.2`, `torchmetrics<0.12.0` — none of which reliably have wheels for very new Python versions. Recreate the conda env with Python 3.10 first if that's your situation:
```bash
conda deactivate
conda env remove -n solo_learn
conda create -n solo_learn python=3.10 -y
conda activate solo_learn
pip install --upgrade pip setuptools wheel
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```
`torch==2.1.2`/`torchvision==0.16.2` is the matched release pair from the same month as `lightning==2.1.2`. cu121 wheels work fine even on newer drivers (e.g. CUDA 12.8) since CUDA is backwards compatible.)

## The actual fix (setup.py)

Delete the `pkg_resources` import, and replace the (previously unused, and actually broken — see note below) local `parse_requirements` helper with a version that just passes each requirements.txt line straight through.

**Before:**
```python
import os

from pkg_resources import parse_requirements
from setuptools import find_packages, setup

KW = ["artificial intelligence", "deep learning", "unsupervised learning", "contrastive learning"]

REQUIREMENTS_FILE = os.path.join(os.path.dirname(__file__), "requirements.txt")
with open(REQUIREMENTS_FILE) as fo:
    REQUIREMENTS = [str(req) for req in parse_requirements(fo.readlines())]

EXTRA_REQUIREMENTS = {
    "dali": ["nvidia-dali-cuda110"],
    "umap": ["matplotlib", "seaborn", "pandas", "umap-learn"],
    "h5": ["h5py"],
}


def parse_requirements(path):
    with open(path) as f:
        requirements = [p.strip().split()[-1] for p in f.readlines()]
    return requirements
```

**After:**
```python
import os

from setuptools import find_packages, setup

KW = ["artificial intelligence", "deep learning", "unsupervised learning", "contrastive learning"]


def parse_requirements(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


REQUIREMENTS_FILE = os.path.join(os.path.dirname(__file__), "requirements.txt")
REQUIREMENTS = parse_requirements(REQUIREMENTS_FILE)

EXTRA_REQUIREMENTS = {
    "dali": ["nvidia-dali-cuda110"],
    "umap": ["matplotlib", "seaborn", "pandas", "umap-learn"],
    "h5": ["h5py"],
}
```

(Rest of `setup.py`, including the `setup(...)` call, is unchanged.)

This drops the `pkg_resources` dependency entirely and just passes each non-blank `requirements.txt` line straight through as a PEP 508 requirement string, which is all `install_requires` needs.

**Note on the original dead-code helper:** the version that used to sit here did `p.strip().split()[-1]`, which looks plausible but is actually broken — it splits each line on whitespace and keeps only the *last* token. That silently truncates any line with an internal space, e.g. `requirements.txt` line 5, `torchmetrics>=0.6.0, <0.12.0`, becomes just `<0.12.0` — losing the package name and making `setup.py` fail with `Expected package name at the start of dependency specifier`. This is almost certainly why that helper was defined but never actually called in the original code. Don't reuse that snippet — pass lines straight through instead.

## Verifying the fix

No need for a full install to sanity-check the parsing logic — this is enough (confirmed working):
```bash
python3 setup.py --name --version
# -> solo-learn
#    1.0.6
# (no pkg_resources warning, no "Expected package name" error)
```

Then the real install:
```bash
pip3 install -e .[umap,h5]
python -c "import solo; import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Note

This is a real upstream bug that will hit anyone installing solo-learn with a modern `setuptools` — worth upstreaming as a small PR.
