import os
import requests
from pathlib import Path
from typing import Optional, Union, List
import ntpath
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def filename_from_path(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


class CsvDataset:
    def __init__(self, names: List[Union[Path, str]]):
        dfs = [pd.read_csv(filename) for filename in names]
        self.assign(*dfs)
        self.filter(*dfs)

    def assign(self, expression_df, batches_df, labels_df):
        # self.labels = LabelEncoder().fit_transform(labels_df.to_numpy())
        labels_np = labels_df.to_numpy()
        self.labels = OneHotEncoder(sparse=False).fit_transform(
            labels_np.reshape(-1, 1)
        )
        self.n_labels = np.unique(labels_np).shape[0]
        # self.batch_indices = LabelEncoder().fit_transform(batches_df.to_numpy())
        batch_np = batches_df.to_numpy()
        self.batch_indices = OneHotEncoder(sparse=False).fit_transform(
            batch_np.reshape(-1, 1)
        )
        self.n_batches = np.unique(batch_np).shape[0]
        self.X = expression_df.to_numpy()
        self.nb_genes = self.X.shape[1]

    def filter(self, expression_df, batches_df, labels_df):
        for checking in (self.labels, self.batch_indices):
            labels, counts = np.unique(checking, return_counts=True)
            for idx, count in enumerate(counts):
                if count < 2:
                    cell_name = labels_df[labels_df.label == labels[idx]].index[0]
                    expression_df = expression_df[expression_df.index != cell_name]
                    labels_df = labels_df[labels_df.index != cell_name]
                    batches_df = batches_df[batches_df.index != cell_name]
        self.assign(expression_df, batches_df, labels_df)

    def __len__(self):
        return self.X.shape[0]

    def split(
        self,
        test_size: float = 0.15,
        for_classifiers: bool = True,
        classif_size: Optional[float] = None,
    ) -> tuple:
        """Split data into train and test sets.
        It optional splits train data into vae train data and classifirs train data
        :param test_size: float part of test size
        :param for_classifiers: bool if True returns data for classifiers training additionally"""
        (
            train_expression,
            test_expression,
            train_batches,
            test_batches,
            train_labels,
            test_labels,
        ) = train_test_split(
            self.X, self.batch_indices, self.labels, test_size=test_size,
            stratify=self.batch_indices
        )

        if for_classifiers:
            (
                train_expression,
                class_expression,
                train_batches,
                class_batches,
                train_labels,
                class_labels,
            ) = train_test_split(
                train_expression, train_batches, train_labels, test_size=classif_size,
                stratify=train_batches
            )
            return (
                (train_expression, train_batches, train_labels),
                (test_expression, test_batches, test_labels),
                (class_expression, class_batches, class_labels)
            )
        else:
            return (
                (train_expression, train_batches, train_labels),
                (test_expression, test_batches, test_labels)
            )


class ScratchDataset(CsvDataset):
    def __init__(
        self, expression: np.ndarray, batch_indices: np.ndarray, labels: np.ndarray
    ):
        self.X = expression
        self.nb_genes = self.X.shape[1]
        self.labels = labels
        if len(labels.shape) == 1:
            self.n_labels = np.unique(labels).shape[0]
        elif len(labels.shape) == 2:
            self.n_labels = labels.shape[1]
        self.batch_indices = batch_indices
        if len(batch_indices.shape) == 1:
            self.n_batches = np.unique(batch_indices).shape[0]
        elif len(batch_indices.shape) == 2:
            self.n_batches = batch_indices.shape[1]

    def __len__(self):
        return self.X.shape[0]


class MouseDataset(CsvDataset):
    """To work with mouse single-cell expression dataset.
    Contains the ways to read/download data and process it

    To download data provide download flag as True
    If you already have data provide a full path for each one

    :param directory: str or pathlib.Path full path to the directory
                        contains data files. If it's None but filenames provided,
                        class tries to find at './' catalog
    :param expression_filename:
    :param batch_filename:
    :param labels_filename:
    :param download: bool, if download flag is True class downloads dataset files.
        If directory is None it downloads to './'
    """

    def __init__(
        self,
        directory: Optional[Union[Path, str]] = None,
        expression_filename: Optional[Union[Path, str]] = None,
        batch_filename: Optional[Union[Path, str]] = None,
        labels_filename: Optional[Union[Path, str]] = None,
        download: bool = False,
        verbose: bool = True,
    ):
        self.cell_attribute_names = {
            "labels",
            "local_vars",
            "local_means",
            "batch_indices",
        }
        if not (
            directory
            or expression_filename
            or batch_filename
            or labels_filename
            or download
        ):
            if not (
                os.path.isfile("./expression.csv")
                and os.path.isfile("./batches.csv")
                and os.path.isfile("./labels.csv")
            ):
                raise ValueError(
                    "There is not enough data to find the dataset. \
                    To download data, set the flag download = True."
                )

        directory = Path(directory) if directory else Path("./")

        if expression_filename is None:
            expression_filename = directory / "expression.csv"
        else:
            if not isinstance(expression_filename, Path):
                expression_filename = Path(expression_filename)
            if not directory in expression_filename.parents:
                expression_filename = directory / expression_filename

        if isinstance(directory, str):
            directory = Path(directory)
        if batch_filename is None:
            batch_filename = directory / "batches.csv"
        else:
            if not isinstance(batch_filename, Path):
                batch_filename = Path(batch_filename)
            if not directory in batch_filename.parents:
                batch_filename = directory / batch_filename

        if labels_filename is None:
            labels_filename = directory / "labels.csv"
        else:
            if not isinstance(labels_filename, Path):
                labels_filename = Path(labels_filename)
            if not directory in labels_filename.parents:
                labels_filename = directory / labels_filename

        if download:
            self.download(
                "https://ndownloader.figshare.com/files/22334787",
                expression_filename,
                verbose,
            )
            self.download(
                "https://ndownloader.figshare.com/files/22332978",
                batch_filename,
                verbose,
            )
            self.download(
                "https://ndownloader.figshare.com/files/22332975",
                labels_filename,
                verbose,
            )

        expression_df = pd.read_csv(expression_filename, index_col=0)

        labels_df = pd.read_csv(labels_filename, index_col=0)
        assert all(expression_df.index == labels_df.index)

        batches_df = pd.read_csv(batch_filename, index_col=0)
        assert all(expression_df.index == batches_df.index)

        dfs = (expression_df, batches_df, labels_df)
        self.assign(*dfs)
        self.filter(*dfs)

    def download(self, url: str, filename: Union[Path, str], verbose: bool) -> None:
        if verbose:
            print(f"Downloading {filename_from_path(filename)}")
        response = requests.get(url, allow_redirects=True, stream=verbose)
        length = response.headers.get("content-length")
        with open(filename, "wb") as file:
            if not length and verbose:
                file.write(response.content)
            else:
                dl = 0
                length = int(length)
                from tqdm import tqdm
                from math import ceil

                for data in tqdm(
                    response.iter_content(chunk_size=4096),
                    total=ceil(length / 4096),
                    position=0,
                    leave=True,
                ):
                    file.write(data)
                    dl += len(data)
