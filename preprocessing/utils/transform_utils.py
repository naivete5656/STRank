# ------------------------------------------------------------------------------
# Copyright (c) 2007 Free Software Foundation, Inc. https://fsf.org/
# Licensed under the GPL-3.0 License.
# The code is based on CLAM.
# (https://github.com/mahmoodlab/CLAM)
# ------------------------------------------------------------------------------

from torchvision import transforms


def get_eval_transforms(mean, std, target_img_size=-1):
    trsforms = []

    if target_img_size > 0:
        trsforms.append(transforms.Resize(target_img_size))
    trsforms.append(transforms.ToTensor())
    trsforms.append(transforms.Normalize(mean, std))
    trsforms = transforms.Compose(trsforms)

    return trsforms
