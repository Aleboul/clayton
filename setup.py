from setuptools import setup, find_packages

requirements = [
    'numpy >=1.14.6',
    'scipy >=1.1.0',
]

setup_requires = [
    'numpy',
    'scipy'
]

setup(
    name = 'clayton',
    version = '0.0.1',  
    description = 'Sampling from copulae',
    long_description=open('README.md', 'r').read(),
    author = 'Alexis Boulin',
    author_email = 'aboulin@unice.fr',
    url = 'https://github.com/Aleboul/clayton',
    download_url = 'https://github.com/Aleboul/clayton/',
    classifiers = [],
    include_package_data=True,
    packages=find_packages()
)
