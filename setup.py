from setuptools import setup, find_packages
from os.path import join, dirname, basename


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


with open('README.rst', 'r') as file:
    readme = file.read()

with open('requirements.txt', 'r') as req_file:
    requirements = req_file.read().split('\n')


setup_requirements = ["pip>=18.1"]

authors = [
    "N. Russkikh",
    "A. Makarov",
    "D. Antonets",
    "D. Shtokalo",
    "Y. Vyatkin"
]

setup(
    author = authors,
    author_email = "makarov.alxr@yandex.ru",
    classifiers = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Natural Language :: Russian",
        "Programming Language :: Python :: 3.7",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    description = "Style transfer variational autoencoder",
    name = "stVAE",
    long_description = readme,
    long_description_content_type = 'text/x-rst',
    python_requires=">=3.6",
    license = "MIT license",
    packages = find_packages(),
    setup_requires = setup_requirements,
    install_requires = parse_requirements('./requirements.txt'),
    version = "0.2.3",
    url = "https://github.com/NRshka/stvae/source",
)
