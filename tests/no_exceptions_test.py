import os
import stvae

# If we're using gpu accelerator
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

dataset = stvae.datasets.MouseDataset(download=True)
# to make metrics better let's filter genes by variance
N_TOP_GENES = 700
dataset = stvae.preprocessing.get_high_variance_genes(dataset, n_genes=N_TOP_GENES)

# split data
train, test, classif = dataset.split(0.15, True, 0.15)

import torch

# the main way to set option: Use stvae.Config dataclass
cfg = stvae.Config()
cfg.count_labels = dataset.n_labels
cfg.count_classes = dataset.n_batches
cfg.use_cuda = False#torch.cuda.is_available()
cfg.input_dim = N_TOP_GENES
cfg.epochs=1
cfg.classifier_epochs=1
model = stvae.stVAE(cfg)

cfg.verbose = 'none' # output
model.train(train, validation_data=None)
metrics = model.test(test, classif)
print(metrics)
