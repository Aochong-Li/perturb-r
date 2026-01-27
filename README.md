## Installation

### uv environment (recommended)
```bash
# Clone with submodules
git clone --recursive https://github.com/Aochong-Li/perturb-r.git
cd perturb-r

# Install dependencies (creates venv automatically)
uv sync
```

### conda environment
```bash
# Clone with submodules
git clone --recursive https://github.com/Aochong-Li/perturb-r.git
cd perturb-r

# Create environment
conda env create -f environment.yml
conda activate rlvr_eval_empire

# Install additional dependencies
pip install -r requirement.txt
```

