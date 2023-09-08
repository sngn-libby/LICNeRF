# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

from typing import *

from src.data.litdata import (
    LitDataBlender,
    LitDataBlenderMultiScale,
    LitDataLF,
    LitDataLLFF,
    LitDataNeRF360V2,
    LitDataRefNeRFReal,
    LitDataShinyBlender,
    LitDataTnT,
)
from src.model.dvgo.model import LitDVGO
from src.model.mipnerf360.model import LitMipNeRF360
from src.model.mipnerf.model import LitMipNeRF
from src.model.nerf.model import LitNeRF
from src.model.nerfpp.model import LitNeRFPP
from src.model.plenoxel.model import LitPlenoxel
from src.model.refnerf.model import LitRefNeRF

from src.model.dvgo.dvgo import DirectVoxGO
from src.model.mipnerf360.model import MipNeRF360
from src.model.mipnerf.model import MipNeRF
from src.model.nerf.model import NeRF
from src.model.nerfpp.model import NeRFPP
from src.model.plenoxel.sparse_grid import SparseGrid
from src.model.refnerf.model import RefNeRF

from src.compressproj.zoo import image_models
from src.compressproj.models.google import JointAutoregressiveHierarchicalPriors
from src.compressproj.models.sensetime import JointCheckerboardHierarchicalPriors

from licnerf.model import (
    MipNeRFTransformedLIC,
)


research_models = {
    "transformed_lic": MipNeRFTransformedLIC,
}

lic_models = {
    "mbt2018": JointAutoregressiveHierarchicalPriors,
    "checkerboard": JointCheckerboardHierarchicalPriors,
}

base_nerf_models = {
    "nerf": NeRF,
    "mipnerf": MipNeRF,
    "mipnerf360": MipNeRF360,
    "plenoxel": SparseGrid,
    "nerfppp": NeRFPP,
    "dvgo": DirectVoxGO,
    "refnerf": RefNeRF,
}

nerf_models = {
    "nerf": LitNeRF,
    "mipnerf": LitMipNeRF,
    "mipnerf360": LitMipNeRF360,
    "plenoxel": LitPlenoxel,
    "nerfppp": LitNeRFPP,
    "dvgo": LitDVGO,
    "refnerf": LitRefNeRF,
    # "nerfusion": LitNeRFusion,
}


def select_research_model(
        research_model_name: str,
        lic_model_name: str = "mbt2018",
        nerf_model_name: str = None,
        quality=4,
        **kwargs
):
    assert research_model_name in research_models
    assert lic_model_name in image_models
    assert nerf_model_name in base_nerf_models

    nerf_model = base_nerf_models[nerf_model_name]()
    lic_model = image_models[lic_model_name](quality=quality)

    return research_models[research_model_name](lic_model=lic_model,
                                                nerf_model=nerf_model,
                                                **kwargs)


def select_nerf_model(
        model_name: str,
):
    if model_name in nerf_models:
        return nerf_models[model_name]()
    else:
        raise f"Unknown model named {model_name}"


def select_dataset(
    dataset_name: str,
    datadir: str,
    scene_name: str,
    # add_noise: float=0,
):
    if dataset_name == "blender":
        data_fun = LitDataBlender
    elif dataset_name == "blender_multiscale":
        data_fun = LitDataBlenderMultiScale
    elif dataset_name == "llff":
        data_fun = LitDataLLFF
    elif dataset_name == "tanks_and_temples":
        data_fun = LitDataTnT
    elif dataset_name == "lf":
        data_fun = LitDataLF
        datadir = "/".join([datadir.rstrip("/"), "lf_data"])
    elif dataset_name == "nerf_360_v2":
        data_fun = LitDataNeRF360V2
    elif dataset_name == "shiny_blender":
        data_fun = LitDataShinyBlender
    elif dataset_name == "refnerf_real":
        data_fun = LitDataRefNeRFReal

    return data_fun(
        datadir=datadir,
        scene_name=scene_name,
        # add_noise=add_noise,
    )


def select_callback(model_name):

    callbacks = []

    if model_name == "plenoxel":
        import src.model.plenoxel.model as model

        callbacks += [model.ResampleCallBack()]

    if model_name == "dvgo":
        import src.model.dvgo.model as model

        callbacks += [
            model.Coarse2Fine(),
            model.ProgressiveScaling(),
            model.UpdateOccupancyMask(),
        ]

    return callbacks
