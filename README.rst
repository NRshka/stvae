==================
Style transfer VAE
==================

.. image:: https://img.shields.io/pypi/v/stvae?color=green
   :target: https://pypi.org/project/stVAE/

.. image:: https://travis-ci.org/NRshka/stvae.svg?branch=master
    :target: https://travis-ci.org/NRshka/stvae

The official pytorch implementation of "*Style transfer with variational autoencoders is a promising approach to RNA-Seq data harmonization and analysis*".
The package contains a code for training and testing the model, as well as a code for working with different types of datasets.

**Installation**

To install the latest version from PyPI, use:

>>> pip install stvae

**Benchmarks**

The original code containing code with testing several models can be found here_.

.. _here: https://github.com/NRshka/stvae-source

**Example**

.. code-block:: python

   ds = stvae.datasets.MouseDataset()
   cfg = stvae.Config()
   train, test, classif = ds.split(0.15, True, 0.15)
   cfg.count_classes = ds.n_labels
   cfg.count_classes = ds.n_batches
   cfg.input_dim = ds.nb_genes
   model = stvae.stVAE(cfg)
   model.train(train, None)
   d = model.test(test, classif)