# ------------------------------------------------------------------------------
# Copyright (c) 2007 Free Software Foundation, Inc. https://fsf.org/
# Licensed under the GPL-3.0 License.
# The code is based on CLAM.
# (https://github.com/mahmoodlab/CLAM)
# ------------------------------------------------------------------------------


import os
from functools import partial
import timm

from .timm_wrapper import TimmCNNEncoder
import torchvision
import torch
import torch.nn as nn
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms


def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = ""
    # check if CONCH_CKPT_PATH is set and conch is installed, catch exception if not
    try:
        from conch.open_clip_custom import create_model_from_pretrained

        # check if CONCH_CKPT_PATH is set
        if "CONCH_CKPT_PATH" not in os.environ:
            raise ValueError("CONCH_CKPT_PATH not set")
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ["CONCH_CKPT_PATH"]
    except Exception as e:
        print(e)
        print("CONCH not installed or CONCH_CKPT_PATH not set")
    return HAS_CONCH, CONCH_CKPT_PATH


def has_UNI():
    HAS_UNI = False
    UNI_CKPT_PATH = ""
    # check if UNI_CKPT_PATH is set, catch exception if not
    try:
        # check if UNI_CKPT_PATH is set
        if "UNI_CKPT_PATH" not in os.environ:
            raise ValueError("UNI_CKPT_PATH not set")
        HAS_UNI = True
        UNI_CKPT_PATH = os.environ["UNI_CKPT_PATH"]
    except Exception as e:
        print(e)
    return HAS_UNI, UNI_CKPT_PATH


def get_encoder(model_name, target_img_size=224):
    img_transforms = None
    print("loading model checkpoint")
    if model_name == "resnet50_trunc":
        model = TimmCNNEncoder()
    if model_name == "densenet121":
        from torchvision import transforms, models
        import torch.nn as nn

        model = torchvision.models.densenet121(pretrained=True)
        model.classifier = nn.Identity()

        img_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  # ImageNet標準  # ImageNet標準
                ),
            ]
        )
    elif model_name == "resnet50":
        from torchvision import transforms, models
        import torch.nn as nn

        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model = nn.Sequential(*list(model.children())[:-2])
        img_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )
    elif model_name == "uni_v1":
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, "UNI is not available"
        model = timm.create_model(
            "vit_large_patch16_224",
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
    elif model_name == "conch_v1":
        HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
        # assert HAS_CONCH, "CONCH is not available"
        from conch.open_clip_custom import create_model_from_pretrained

        if HAS_CONCH:
            model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        else:
            model, preprocess = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch")
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    elif model_name == "hopt":
        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
    elif model_name == "dinov2":
        model = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        data_config = timm.data.resolve_model_data_config(model)
        img_transforms = timm.data.create_transform(**data_config, is_training=False)
    else:
        raise NotImplementedError("model {} not implemented".format(model_name))

    print(model)
    if img_transforms is None:
        constants = MODEL2CONSTANTS[model_name]
        img_transforms = get_eval_transforms(
            mean=constants["mean"],
            std=constants["std"],
            target_img_size=target_img_size,
        )

    return model, img_transforms
