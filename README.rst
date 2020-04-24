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

   ds = stvae.datasets.MouseDataset(download=True) # download data to the current directory
   cfg = stvae.Config()
   train, test, classif = ds.split(0.15, True, 0.15)
   cfg.count_classes = ds.n_labels
   cfg.count_classes = ds.n_batches
   cfg.input_dim = ds.nb_genes
   cfg.use_cuda = True # if you have a CUDA compatibility gpu
   cfg.epochs = 600 # number of training epocs
   cfg.classifier_epochs = 450 # number of epochs for testing classifirs training
   model = stvae.stVAE(cfg)
   model.train(train, None)
   d = model.test(test, classif)