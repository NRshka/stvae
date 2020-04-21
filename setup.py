from setuptools import setup, find_packages
from os.path import join, dirname, basename


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
    version = "0.2.2",
    url = "https://github.com/NRshka/stvae/source"
)
