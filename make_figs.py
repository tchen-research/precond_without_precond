import subprocess

files = [
    'intro',
    'iter_error',
    'mu_error',
    'sqrt_iter_error',
    'iter_error_fp_compare',
    'iter_error_matvec',
    'iter_error_fp',
    'blocksize_error',
    'blocksize_condno',
    'spectrums',
]

for file in files:
    subprocess.run(f'jupyter nbconvert --execute --to notebook --inplace --allow-errors --ExecutePreprocessor.timeout=-1 {file}.ipynb',shell=True)