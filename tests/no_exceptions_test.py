import os
import stvae

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ds = stvae.datasets.MouseDataset()
cfg = stvae.Config()
train, test, classif = ds.split(0.15, True, 0.15)

cfg.count_classes = ds.n_labels
cfg.count_classes = ds.n_batches
cfg.input_dim = ds.nb_genes
cfg.epochs = 1
cfg.classifier_epochs = 1
model = stvae.stVAE(cfg)
model.train(train, None)

d = model.test(test, classif)
