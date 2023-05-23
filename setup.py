from setuptools import setup, find_packages

setup(
   name='pod2seetings',
   packages=find_packages(),
   version='0.1',
   description='Optimization of experimental settings guided by POD analysis',
   author='V. Andriiashen',
   author_email='vladyslav.andriiashen@cwi.nl',
   url = 'https://github.com/vandriiashen/pod2settings',
   package_dir={'pod2settings': 'pod2settings'}
)
