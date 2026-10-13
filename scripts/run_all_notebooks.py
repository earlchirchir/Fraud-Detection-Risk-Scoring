"""
Master pipeline runner executing all 6 notebooks in isolated processes.
"""

import os
import sys
import subprocess
import time

notebooks = [
    ('build_nb01.py', '01_data_cleaning.ipynb'),
    ('build_nb02.py', '02_EDA.ipynb'),
    ('build_nb03.py', '03_feature_engineering.ipynb'),
    ('build_nb04.py', '04_model_development.ipynb'),
    ('build_nb05.py', '05_model_evaluation_and_thresholds.ipynb'),
    ('build_nb06.py', '06_production_governance.ipynb')
]

runner_script = """
import sys
import nbformat
from nbclient import NotebookClient

nb_path = sys.argv[1]
print(f'Executing {nb_path} in isolated process...')

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=1200, kernel_name='python3', resources={'metadata': {'path': 'notebooks/'}})
client.execute()

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f'SUCCESS: {nb_path} executed and saved with outputs.')
"""

with open(os.path.join('scripts', 'exec_single_nb.py'), 'w', encoding='utf-8') as f:
    f.write(runner_script)

def run_all():
    start_total = time.time()
    for build_script, nb_file in notebooks:
        print(f"\n------------------------------------------")
        print(f"STEP: {build_script} -> {nb_file}")
        print(f"------------------------------------------")
        
        # 1. Build notebook
        subprocess.run([sys.executable, os.path.join('scripts', build_script)], check=True)
        
        # 2. Execute notebook in isolated process
        nb_path = os.path.join('notebooks', nb_file)
        t0 = time.time()
        res = subprocess.run([sys.executable, os.path.join('scripts', 'exec_single_nb.py'), nb_path], capture_output=True, text=True)
        dur = time.time() - t0
        
        if res.returncode != 0:
            print(f"ERROR executing {nb_file}:")
            print(res.stdout)
            print(res.stderr)
            sys.exit(1)
        else:
            print(res.stdout.strip())
            print(f"Completed in {dur:.1f}s.")

    print(f"\n[DONE] ALL 6 NOTEBOOKS COMPLETED IN {time.time() - start_total:.1f}s.")

if __name__ == '__main__':
    run_all()
