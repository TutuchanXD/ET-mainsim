"""One-shot CI bootstrap. The full suite remains ci.run_full_pytest, unfiltered."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def hermetic_environment(work: Path, base=None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'ET_DATA_DIR',
                'GAIA_CATALOG_DIR', 'ET_FOCALPLANE_ROOT', 'RESULTS_ROOT'):
        env.pop(key, None)
    env.update(ET_DATA_DIR=str(work / 'missing-data'), CUDA_VISIBLE_DEVICES='',
               PYTEST_ADDOPTS='', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
               MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1',
               PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_CONFIG_FILE=os.devnull)
    return env


def install_plan(python: Path, root: Path, coordinate: Path, photsim: Path):
    pip = [str(python), '-m', 'pip']
    return [
        pip + ['install', '--upgrade', 'pip'],
        pip + ['install', '--index-url', 'https://download.pytorch.org/whl/cpu', 'torch>=2.7,<3'],
        pip + ['install', str(coordinate)],
        pip + ['install', str(photsim) + '[gpu]'],
        pip + ['install', '-e', str(root) + '[test,release]'],
        pip + ['check'],
    ]


def run(command, *, env, cwd=ROOT):
    subprocess.run([str(part) for part in command], cwd=cwd, env=env, check=True)


def dependency(name: str, sha: str, work: Path, env):
    existing = ROOT / '.ci-dependencies' / name
    target = existing if existing.exists() else work / name
    if not existing.exists():
        url = (f'git@github.com:TutuchanXD/{name}.git' if name == 'Photsim7'
               else f'https://github.com/TutuchanXD/{name}.git')
        run(['git', 'init', target], env=env)
        run(['git', '-C', target, 'fetch', '--depth=1', url, sha], env=env)
        run(['git', '-C', target, 'checkout', '--detach', 'FETCH_HEAD'], env=env)
    observed = subprocess.check_output(['git', '-C', target, 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', target, 'status', '--porcelain', '--untracked-files=no'], text=True)
    if observed != sha or dirty:
        raise ValueError(f'{name} must be a clean checkout of frozen commit {sha}')
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('smoke', 'full'))
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--output', type=Path, default=ROOT / '.ci-results')
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='et-mainsim-ci-') as temporary:
        work = Path(temporary)
        env = hermetic_environment(work)
        run([args.python, '-m', 'venv', work / 'env'], env=env)
        python = work / 'env/bin/python'
        version = subprocess.check_output([python, '-c', 'import sys; print("%d.%d" % sys.version_info[:2])'], text=True).strip()
        if version not in ('3.12', '3.13'):
            raise ValueError('CI requires Python 3.12 or 3.13')
        env['PATH'] = str(python.parent) + os.pathsep + env.get('PATH', '')
        run([python, '-m', 'pip', 'install', 'PyYAML==6.0.3'], env=env)
        run([python, '-m', 'ci.verify_full_test_workflow'], env=env)
        if args.mode == 'smoke':
            run([python, '-m', 'pip', 'install', '--upgrade', 'pip', 'build'], env=env)
            run([python, '-m', 'build', '--outdir', work / 'dist'], env=env)
            wheel, = (work / 'dist').glob('*.whl')
            run([python, '-m', 'pip', 'install', '--no-deps', wheel], env=env)
            run([python, '-I', '-c', "import et_mainsim; assert et_mainsim.__version__ == '0.1.0'"], env=env, cwd=work)
            run([python.parent / 'et-mainsim', '--version'], env=env, cwd=work)
            run([python.parent / 'et-mainsim', '--help'], env=env, cwd=work)
            run([python, '-m', 'compileall', '-q', 'src'], env=env)
        else:
            with (ROOT / 'ci/full_pytest_contract.toml').open('rb') as stream:
                contract = tomllib.load(stream)
            coordinate = dependency('ET-coordinate', contract['dependencies']['et_coordinate_commit'], work, env)
            photsim = dependency('Photsim7', contract['dependencies']['photsim7_commit'], work, env)
            for command in install_plan(python, ROOT, coordinate, photsim):
                run(command, env=env)
            # Explicit caller path supports the unchanged Actions artifact contract.
            env['FULL_PYTEST_RECEIPT'] = os.environ.get('FULL_PYTEST_RECEIPT', str(args.output / f'full-pytest-receipt-py{version}.json'))
            run([python, '-m', 'ci.run_full_pytest'], env=env)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
