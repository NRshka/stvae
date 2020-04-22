from typing import Sized, Callable, Union, Optional
from warnings import warn
import numpy as np
from numpy import array as Array
from torch import Tensor
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader as TorchDataLoader
from scanpy import AnnData

from .model import VAE, Latent_discriminator
from .config import Config
from .datasets import CsvDataset
from .train_script import train
from .test_script import test, train_classifiers


def _cast_data(data, cfg: Config, mode: str, datatype: str = "", preproc: bool = True):
    assert mode in ("train", "test")

    kwargs = {"batch_size": cfg.batch_size}  # for torch DataLoader
    if cfg.num_workers:
        kwargs["num_workers"] = cfg.num_workers

    if isinstance(data, (CsvDataset)):  # TODO scvi.dataset.GeneExpression
        expression = data.X
        if preproc:
            expression = np.log(expression + 1.)
        tensors = (Tensor(expression), Tensor(data.batch_indices))
        if mode == "test":
            tensors += (Tensor(data.labels),)
        ds = TensorDataset(*tensors)
        return TorchDataLoader(ds, **kwargs)
    elif isinstance(data, (tuple, list, Array)):
        items_shape = data[0].shape
        for item in data:
            if item.shape[0] != items_shape[0]:
                raise ValueError(f"Size mismatch {items_shape[0]} and {item.shape[0]}")

        '''
        if (len(data) != 2 and mode == "train") or (len(data) != 3 and mode == "test"):
            raise ValueError(
                f"Expected {datatype} data \
                with dim=2, got dim={len(data)}"
            )
        '''

        n = 3 if mode == "test" else 2
        tensors = [Tensor(data[i]) for i in range(n)]
        if preproc:
            tensors[0] = np.log(tensors[0] + 1.)
        ds = TensorDataset(*tensors)

        return TorchDataLoader(ds, **kwargs)
    elif isinstance(data, dict):
        if "expression" in data.keys():
            expression = data["expression"]
        elif "X" in data.keys():
            expression = data["X"]
        else:
            raise KeyError(f"{datatype} data must contains 'X' or 'expression' key")

        if preproc:
            expression = np.log(expression + 1.)

        try:
            batches = data["batch_indices"]
            labels = data["labels"]
        except KeyError as err:
            raise KeyError(f"{datatype} data must contains {str(err)} key")

        tensors = (expression, batches, labels)
        ds = TensorDataset(*tensors)
        return TorchDataLoader(
            ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers
        )
    elif isinstance(data, AnnData):
        raise NotImplementedError()
    elif isinstance(data, TorchDataLoader):
        return data
    elif isinstance(data, TensorDataset):
        return TorchDataLoader(
            ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers
        )
    else:
        raise NotImplementedError()


class stVAE:
    def __init__(self, cfg: Config):
        self.model = VAE(
            bottleneck=cfg.bottleneck,
            input_size=cfg.input_dim,
            count_classes=cfg.count_classes,
            n_layers=cfg.n_layers,
            scale_alpha=cfg.scale_alpha,
            ohe_latent_dim=cfg.condition_latent_dim,
        )
        self.disc = Latent_discriminator(cfg.bottleneck, cfg.count_classes)
        self.cfg = cfg
        self.celltype_clf = None
        self.form_clf = None
        if cfg.use_cuda:
            self.model = self.model.cuda()
            self.disc = self.disc.cuda()

    def train(
        self,
        train_data: Union[Sized, AnnData],
        validation_data: Optional[Union[Sized, AnnData]],
        random_seed=0,
        preprocess: bool = True
    ) -> None:
        train_data = _cast_data(train_data, self.cfg, "train", preproc=preprocess)
        valid_data = None
        if not validation_data is None:
            valid_data = _cast_data(validation_data, self.cfg, "train", "validation")
        train(train_data, valid_data, self.cfg, self.model, self.disc)

    def train_classifiers(self, train_data) -> None:
        train_data = _cast_data(train_data, self.cfg, "train")
        self.celltype_clf, self.form_clf = train_classifiers(
            self.cfg, train_data, self.cfg.count_labels, self.cfg.count_classes
        )

    def test(
        self,
        test_data,
        classifier_train_data=None,
        custom_metrics=[],
        overwrite: Union[bool] = False,
        preprocess: bool = True
    ) -> dict:
        if not classifier_train_data is None:
            classifier_train_data = _cast_data(classifier_train_data, self.cfg, "test", preprocess)
        test_data = _cast_data(test_data, self.cfg, "test")
        expression = test_data.dataset.tensors[0].numpy()
        batch_indices = test_data.dataset.tensors[1].numpy()
        labels = test_data.dataset.tensors[2].numpy()
        if self.celltype_clf is None or self.form_clf is None:
            if classifier_train_data is None:
                raise ValueError(
                    "If classifiers are not trained, \
                    you must provide data for training."
                )

            return test(
                self.cfg,
                self.model,
                self.disc,
                classifier_train_data,
                expression,
                batch_indices,
                labels,
            )
        else:
            if not classifier_train_data is None:
                warn(
                    'Data is specified, but classifiers already exist. \
                    New classifiers will be trained, but they will not be saved. \
                        To save new classifiers, set the flag "overwrite=True'
                )

                return test(
                    self.cfg,
                    self.model,
                    self.disc,
                    classifier_train_data,
                    expression,
                    batch_indices,
                    labels,
                )
            else:
                return test(
                    self.cfg,
                    self.model,
                    self.disc,
                    None,
                    expression,
                    batch_indices,
                    labels,
                )

    def set_config(self, cfg: Config) -> None:
        self.cfg = cfg

    def __repr__(self) -> str:
        repr = (
            "stVAE class contains\n"
            + "VAE:\n"
            + self.model.__repr__()
            + "\nDiscriminator:\n"
            + self.disc.__repr__()
            + "\nAnd configuration file "
        )
        if "name" in self.cfg.__dict__:
            repr += self.cfg.name
        return repr
