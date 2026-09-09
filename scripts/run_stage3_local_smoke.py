#!/usr/bin/env python3
"""Run unchanged smoke with an untracked root Finder file temporarily preserved.

Never hide tracked changes or relax the upstream clean-check assertion.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curv-root', type=Path, default=ROOT.parent / 'CURV')
    args = parser.parse_args()
    upstream = args.curv_root.resolve()
    finder = upstream / '.DS_Store'
    tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', '.DS_Store'],
                             cwd=upstream, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL).returncode == 0
    holding = None
    if finder.is_file() and not finder.is_symlink() and not tracked:
        holding = Path(tempfile.mkdtemp(prefix='curv-finder-preserved-'))
        shutil.move(str(finder), str(holding / '.DS_Store'))
        print('Temporarily preserving untracked Finder metadata:', holding, flush=True)
    try:
        environment = os.environ.copy()
        environment.setdefault('HF_HUB_OFFLINE', '1')
        environment.setdefault('TRANSFORMERS_OFFLINE', '1')
        subprocess.run(['bash', str(ROOT / 'scripts/run_stage3_smoke.sh'),
                        '--curv-root', str(upstream)], cwd=ROOT, env=environment, check=True)
    finally:
        if holding:
            if finder.exists() or finder.is_symlink():
                # Preserve a new Finder file too, instead of overwriting it.
                shutil.move(str(finder), str(holding / 'regenerated.DS_Store'))
            shutil.move(str(holding / '.DS_Store'), str(finder))
            print('Original Finder metadata restored:', finder, flush=True)
            if any(holding.iterdir()):
                print('Regenerated Finder metadata preserved at:', holding, flush=True)
            else:
                holding.rmdir()


if __name__ == '__main__':
    main()
