from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setup(name='celtic',
      description='Single cell in silico labeling model using tabular input context augmentation.',
      author='Nitsan Elmalam',
      author_email='enitsan8@gmail.com',
      url='https://github.com/zaritskylab/CELTIC',
      packages=find_packages(),
      python_requires='>=3.7.12', # test 3.11
      install_requires=required,
      version='1.0.0')