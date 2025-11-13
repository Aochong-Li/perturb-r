## Installation

### conda environment
```bash
conda env create -f environment.yml
pip install --no-deps requirements.txt
```


### uv environment
```bash
uv venv rlvr_eval_empire --python 3.10.16
source rlvr_eval_empire/bin/activate

uv pip install --no-deps -r requirements.txt
uv pip install --no-deps latex2sympy2==1.9.1
```
