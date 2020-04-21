from typing import Optional, SIzed
import numpy as np
import scanpy as sc
from scvi.dataset import GeneExpressionDataset

from datasets import ScratchDataset
from .utils import make_anndata, scvi_anndata


def predefined_preprocessing(
    data: Sized, framework: str, data_format: Optional[str] = "adaptive"
):
    """Prepare for any supported frameworks
    :Param data: data of supoorted type [scanpy.Anndata,
        tuple(at least one numpy.ndarray), scVI dataset like class]
    :Param framework: flag defining one of supported frameworks
    :Param data_format: define what kind of data is passed
    Note: Unrecognized framework param occurs NotImplementedError
        Unrecognized data_format param occurs ValueError
        If type of data unrecognized it occurs TypeError
    """

    _supported_formats = ["adaptive", "raw", "scvi", "anndata"]
    _supported_frameworks = ["stvae", "scvi", "scgen", "trvae"]
    framework = framework.lower()

    if not framework in _supported_frameworks:
        raise NotImplementedError(
            f"{framework} isn't supported value \
                                    {' '.join(_supported_frameworks)}"
        )
    if not data_format in _supported_formats:
        raise ValueError(
            f"{data_format} is not supported. \
                            Supported values are {''.join(_supported_formats)}"
        )

    def scvi_format(data, framework):
        if framework == "scvi":
            return data
        elif framework == "stvae":
            expression = np.log(data.X + 1.0)
            return ScratchDataset(expression, data.batch_indices, data.labels)
        data = scvi_anndata(data)  # trvae and scgen
        return data

    def raw_format(data, framework):
        if framework == "scvi":
            pass  # find a way to make scvi dataset from scratch by native classes
        elif framework == "stvae":
            if len(data) != 3:
                raise ValueError(f"Exprected data len = 3, got {len(data)}")
            expression = np.log(data[0] + 1.0)
            return ScratchDataset(
                expression, data[1], data[2]  # expression, batch indices, label indices
            )
        elif framework in ("scgen", "trvae"):
            if len(data) < 1:
                raise ValueError("Data must not be empty")
            _keys = ("batch_index", "cell_info", "variables_info")
            anndata_arguments = {_keys[i]: data[i + 1] for i in range(len(data))}
            if len(data) > 1:
                anndata_arguments["batch_name"] = "condition"
            if len(data) > 2:
                anndata_arguments["cell_info_name"] = "cell_type"

            return make_anndata(data[0], **anndata_arguments)

    if data_format == "scvi":
        data = scvi_format(data, framework)
    elif data_format == "raw":
        data = raw_format(data, framework)
    # do anndata
    elif data_format == "adaptive":
        # Attention: GeneExpressionDataset is Sized
        if isinstance(data, GeneExpressionDataset):
            data = scvi_format(data, framework)
        elif isinstance(data, Iterable):
            data = raw_format(data, framework)
        else:
            raise TypeError(f"Unrecognized data type: {type(data)}")
        # do AnnData
    if framework in ("scgen", "trvae"):
        sc.pp.normalize_per_cell(data)
        sc.pp.log1p(data)

    return data