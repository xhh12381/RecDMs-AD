import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
from torchvision import transforms
import imgaug.augmenters as iaa
import PIL
from perlin import rand_perlin_2d_np


class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir + "/*/*.png"))
        self.resize_shape = resize_shape
        self.transform_img = [
            transforms.Resize(resize_shape),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(resize_shape),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if mask_path is not None:
            mask = PIL.Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # if mask_path is not None:
        #     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # else:
        #     mask = np.zeros((image.shape[0], image.shape[1]))
        # if self.resize_shape is not None:
        #     image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        #     mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))
        #
        # image = image / 255.0
        # mask = mask / 255.0
        #
        # image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        # mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)
        #
        # image = np.transpose(image, (2, 0, 1))
        # mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir in ['good']:
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            if len(file_name.split("_aug.")) > 1:
                mask_file_name = file_name.split("_aug.")[0] + "_mask.png"
            else:
                mask_file_name = file_name.split(".")[0] + "_mask.png"
            # mask_file_name = file_name.split("_aug.")[0] + "_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

            rec_path = os.path.join(dir_path, '../../rec/')
            rec_path = os.path.join(rec_path, base_dir)
            if len(file_name.split("_aug.")) > 1:
                rec_file_name = file_name.split("_aug.")[0] + "_rec.png"
            else:
                rec_file_name = file_name.split(".")[0] + "_rec.png"
            # rec_file_name = file_name.split("_aug.")[0] + "_rec.png"
            rec_path = os.path.join(rec_path, rec_file_name)
            rec = PIL.Image.open(rec_path).convert("RGB")
            rec = self.transform_img(rec)

        # sample = {'image': image, 'has_anomaly': has_anomaly, 'mask': mask, 'idx': idx}
        sample = {'image': image, 'has_anomaly': has_anomaly, 'mask': mask, 'rec': rec, 'idx': idx}

        return sample


class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self, root_dir, anomaly_source_path, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape = resize_shape

        self.image_paths = sorted(glob.glob(root_dir + "/*.png"))

        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path + "/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def __len__(self):
        return len(self.image_paths)

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        # aug = iaa.Sequential([self.augmenters[aug_ind[0]],
        #                       self.augmenters[aug_ind[1]],
        #                       self.augmenters[aug_ind[2]]])
        aug = iaa.Sequential([])
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.cvtColor(anomaly_source_img, cv2.COLOR_BGR2RGB)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.6
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * perlin_thr

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 1.0:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0], dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = perlin_thr.astype(np.float32)
            augmented_image = msk * augmented_image + (1 - msk) * image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly = 0.0
            return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        # do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        # if do_aug_orig:
        #     image = self.rot(image=image)

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        # idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx],
                                                                                 self.anomaly_source_paths[
                                                                                     anomaly_source_idx])
        sample = {'image': image, "mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample
