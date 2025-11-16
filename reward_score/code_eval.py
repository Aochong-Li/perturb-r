# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import pandas as pd
import os
import re
import contextlib
import faulthandler
import io
import multiprocessing
import platform
import signal
import tempfile
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def extract_code(text: str) -> Optional[str]:
    """Extract code from ```python ``` blocks.

    Args:
        text: Text potentially containing code blocks

    Returns:
        Extracted code or None if no code block found
    """
    if not text:
        return None

    # Match ```python ... ``` or ```\n...\n```
    pattern = r'```(?:python)?\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        # Return the last code block (most likely to be the final answer)
        return matches[-1].strip()

    # If no code block found, return the text as-is
    return text.strip()


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""
    def read(self, *args, **kwargs):
        raise IOError
    def readline(self, *args, **kwargs):
        raise IOError
    def readlines(self, *args, **kwargs):
        raise IOError
    def readable(self, *args, **kwargs):
        return False


class redirect_stdin(contextlib._RedirectStream):
    _stream = "stdin"


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """Disable destructive functions to prevent generated code from interfering with tests."""
    if maximum_memory_bytes is not None:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins
    builtins.exit = None
    builtins.quit = None

    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess
    subprocess.Popen = None

    # __builtins__ can be either a dict or a module depending on context
    if isinstance(__builtins__, dict):
        __builtins__["help"] = None
    else:
        __builtins__.help = None

    import sys
    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def unsafe_execute(code: str, timeout: float, result):
    """Execute code in a sandboxed environment."""
    with create_tempdir():
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        reliability_guard()

        try:
            exec_globals = {}
            with swallow_io():
                with time_limit(timeout):
                    exec(code, exec_globals)
            result.append("passed")
        except TimeoutException:
            result.append("timed out")
        except BaseException as e:
            result.append(f"failed: {e}")

        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


def check_code_correctness(code: str, timeout: float = 3.0) -> bool:
    """Check if code executes without errors.

    Args:
        code: Code to execute
        timeout: Timeout in seconds

    Returns:
        True if code executes successfully, False otherwise
    """
    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute, args=(code, timeout, result))
    p.start()
    p.join(timeout=timeout + 1)

    if p.is_alive():
        p.kill()

    if not result:
        result.append("timed out")

    return result[0] == "passed"


def normalize_assert(text: str) -> str:
    """Normalize assert statement for comparison."""
    if not text:
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    return text


def evaluate_humaneval(pred: str, solution: str, entry_point: str) -> Optional[float]:
    """Evaluate HumanEval and HumanEval Plus predictions."""
    code = extract_code(pred)
    if not code:
        return 0.0

    # Construct test program: extracted_code + solution + check(entry_point)
    test_program = f"{code}\n\n{solution}\n\ncheck({entry_point})"

    try:
        passed = check_code_correctness(test_program)
        return 1.0 if passed else 0.0
    except Exception:
        return 0.0


def evaluate_mbpp(pred: str, solution: str) -> Optional[float]:
    """Evaluate MBPP and MBPP Plus predictions."""
    code = extract_code(pred)
    if not code:
        return 0.0

    # Construct test program: extracted_code + solution (direct asserts)
    test_program = f"{code}\n\n{solution}"

    try:
        passed = check_code_correctness(test_program)
        return 1.0 if passed else 0.0
    except Exception:
        return 0.0


def evaluate_cruxeval(problem: str, pred: str, solution: str) -> Optional[float]:
    """Evaluate CruxEval predictions using dual approach.

    Try both execution and string matching - pass if either succeeds.
    """
    # Approach A: Try to execute the predicted code
    code = extract_code(pred)
    if code:
        try:
            text = problem.split('```python')[-1].replace('```\n[/PYTHON]','')
            func = re.sub(r"^assert.*\?$", "", text, flags=re.MULTILINE)
            assert_stmt = re.findall(r"^assert.*", code, flags=re.MULTILINE)
            assert_stmt = assert_stmt[-1]
            test_program = f"{func}\n\n{assert_stmt}"

            passed = check_code_correctness(test_program)
            if passed:
                return 1.0
        except Exception:
            pass

    # Approach B: String comparison of assert statements
    # Extract assert from pred
    pred_assert = None
    if code and 'assert' in code:
        # Extract assert line
        for line in code.split('\n'):
            if line.strip().startswith('assert'):
                pred_assert = line.strip()
                break

    if pred_assert:
        normalized_pred = normalize_assert(pred_assert)
        normalized_solution = normalize_assert(solution)

        if normalized_pred == normalized_solution:
            return 1.0

    return 0.0

def evaluate_row(args):
    """Worker function for parallel evaluation.

    Args:
        args: Tuple of (index, row_dict) where row_dict contains all needed fields

    Returns:
        Tuple of (index, result)
    """
    idx, row = args
    if "distractor_problem" not in row.keys():
        # Benchmark Evaluation
        result = code_verify_score(
            row.get('problem'),
            row.get('pred'),
            row.get('solution'),
            row.get('source'),
            row.get('entry_point')
        )
        return idx, result
    else:
        # Recoverability Evaluation
        if 'humaneval' in row.get('source'):
            problem = row.get('problem')
            m = re.search(r"function name\s+([A-Za-z_]\w*)\s+as entry point", problem)
            entry_point = m.group(1) if m else None
        else:
            entry_point = None

        result = code_verify_score(
            row.get('problem'),
            row.get('pred'),
            row.get('solution'),
            row.get('source'),
            row.get('entry_point', entry_point)
        )
        return idx, result


def code_verify_score(problem: str, pred: str, solution: str, source: str, entry_point: str = None) -> Optional[float]:
    """Verify code correctness based on source type.

    Args:
        pred: Model prediction
        solution: Ground truth test cases
        source: Source dataset (humaneval, mbpp, cruxeval, etc.)
        entry_point: Function entry point (for humaneval)
    Returns:
        1.0 if correct, 0.0 if incorrect, None if unable to evaluate
    """
    if not pred or not solution:
        return None

    try:
        if source in ['humaneval', 'humanevalplus']:
            return evaluate_humaneval(pred, solution, entry_point)
        elif source in ['mbpp', 'mbppplus']:
            return evaluate_mbpp(pred, solution)
        elif source == 'cruxeval':
            return evaluate_cruxeval(problem, pred, solution)
        else:
            return None
    except Exception as e:
        print(f"Error evaluating {source}: {e}")
        return 0.0


if __name__ == "__main__":
    """
    Usage:
    conda activate rlvr_eval_empire
    python reward_score/code_eval.py --input_dir ./results/allcode/benchmark --n_workers 64
    python reward_score/code_eval.py --input_dir ./results/allcode/inject_distractor --n_workers 64
    python reward_score/code_eval.py --file_path ./results/allcode/benchmark/R1-Distill-Qwen-1.5B.pickle --n_workers 64
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=False)
    parser.add_argument("--input_dir", type=str, required=False)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n_workers", type=int, default=32, help="Number of parallel workers")
    args = parser.parse_args()

    if args.file_path:
        df = pd.read_pickle(args.file_path)

        # Parallel evaluation
        print(f"Evaluating {len(df)} samples with {args.n_workers} workers...")
        results = {}

        # Convert dataframe to list of (index, row_dict) tuples for parallel processing
        rows_to_evaluate = [(idx, row.to_dict()) for idx, row in df.iterrows()]

        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = [executor.submit(evaluate_row, row_data) for row_data in rows_to_evaluate]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                idx, result = future.result()
                results[idx] = result

        # Assign results back to dataframe in original order
        df["model_is_correct"] = [results[idx] for idx in df.index]
        df.to_pickle(args.file_path)

        # Print summary statistics
        accuracy = df["model_is_correct"].mean()
        total = len(df)
        correct = df["model_is_correct"].sum()
        print(f"Accuracy: {accuracy:.2%} ({int(correct)}/{total})")

    else:
        for fname in os.listdir(args.input_dir):
            if fname.endswith(".pickle"):
                print(f"\nEvaluating {fname}")
                file_path = os.path.join(args.input_dir, fname)
                df = pd.read_pickle(file_path)

                if "model_is_correct" in df.columns and not args.overwrite:
                    print(f"Skipping {fname} - already has model_is_correct column")
                    continue

                # Parallel evaluation
                print(f"Processing {len(df)} samples with {args.n_workers} workers...")
                results = {}

                # Convert dataframe to list of (index, row_dict) tuples for parallel processing
                rows_to_evaluate = [(idx, row.to_dict()) for idx, row in df.iterrows()]

                with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
                    futures = [executor.submit(evaluate_row, row_data) for row_data in rows_to_evaluate]

                    for future in tqdm(as_completed(futures), total=len(futures), desc=f"  {fname}"):
                        idx, result = future.result()
                        results[idx] = result

                # Assign results back to dataframe in original order
                df["model_is_correct"] = [results[idx] for idx in df.index]
                df.to_pickle(file_path)

                # Print summary statistics by source
                print(f"\n{fname} Results:")
                for source in df['source'].unique():
                    source_df = df[df['source'] == source]
                    accuracy = source_df["model_is_correct"].mean()
                    total = len(source_df)
                    correct = source_df["model_is_correct"].sum()
                    nan_count = source_df["model_is_correct"].isna().sum()
                    print(f"  {source}: {accuracy:.2%} ({int(correct)}/{total}) - NaN: {nan_count}")

                # Overall accuracy
                accuracy = df["model_is_correct"].mean()
                total = len(df)
                correct = df["model_is_correct"].sum()
                print(f"  Overall: {accuracy:.2%} ({int(correct)}/{total})\n")
