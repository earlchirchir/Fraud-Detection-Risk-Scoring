
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
