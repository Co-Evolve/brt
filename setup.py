from setuptools import find_packages, setup

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
        name='biorobot',
        version='0.1.8',
        description='The Bio-inspired Robotics Testbed.',
        long_description=readme,
        url='https://github.com/Co-Evolve/brb',
        license=license,
        packages=find_packages(exclude=('tests', 'docs')),
        install_requires=required
        )
