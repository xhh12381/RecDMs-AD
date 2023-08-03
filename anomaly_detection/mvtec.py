import os
from enum import Enum

import PIL
import torch
from torchvision import transforms

_CLASSNAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MVTecDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
            self,
            source,
            classname,
            imgsize=256,
            split="train",  # train or test
            **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            imgsize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [str]. Indicates if training or test split of the
                   data should be used. Note that test will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(imgsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(imgsize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imgsize, imgsize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        # classname, anomaly, image_path, mask_path, gen_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == "test" and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        # gen = PIL.Image.open(gen_path).convert("RGB")
        # gen = self.transform_img(gen)

        return {
            "image": image,
            "mask": mask,
            # "gen": gen,
            "classname": classname,
            "anomaly": anomaly,  # 类别
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}
        # genpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            # genpath = os.path.join(self.source, classname, "gen")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}
            # genpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                if anomaly not in ["good", "good_train"]:
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                else:
                    maskpaths_per_class[classname]["good"] = None

                # anomaly_gen_path = os.path.join(genpath, anomaly)
                # anomaly_gen_files = sorted(os.listdir(anomaly_gen_path))
                # genpaths_per_class[classname][anomaly] = [
                #     os.path.join(anomaly_gen_path, x) for x in anomaly_gen_files
                # ]

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if anomaly not in ["good", "good_train"]:
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    # data_tuple.append(genpaths_per_class[classname][anomaly][i])
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate
