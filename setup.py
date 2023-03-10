from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='learn-ml',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
)