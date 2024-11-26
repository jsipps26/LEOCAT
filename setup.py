
from setuptools import setup, find_packages

def read_requirements():
	with open('requirements.txt') as fp:
		return fp.read().splitlines()

setup(
	name='leocat',
	version='0.1',
	packages=find_packages(),
	install_requires=read_requirements(),
)
