import glob
import os.path
import random
import shutil

import torch

#
# data_root = "datasets"
# data_classes = os.listdir(data_root)
# for data_class in data_classes:
#     train_imgs = glob.glob(os.path.join(data_root, data_class, "train", data_class, "images", "*.png"))
#     train_img_num = len(train_imgs)
#     test_imgs = glob.glob(os.path.join(data_root, data_class, "test", data_class, "images", "*.png"))
#     test_img_num = len(test_imgs)
#     # img_path = os.path.join(data_root, data_class, "train", data_class, "images")
#     # rec_path = os.path.join(data_root, data_class, "train", "rec", "images")
#     # mask_path = os.path.join(data_root, data_class, "train", "ground_truth", "images")
#     # for i in range(test_img_num // 2):
#     #     idx = train_img_num - (i + 1)
#     #     os.remove(os.path.join(img_path, f"{idx}.png"))
#     #     os.remove(os.path.join(rec_path, f"{idx}_rec.png"))
#     #     os.remove(os.path.join(mask_path, f"{idx}_mask.png"))
#
#     sample_data = random.sample(test_imgs, test_img_num // 4)
#     for idx, img in enumerate(sample_data):
#         rec = img.replace(data_class, "rec").replace("rec", data_class, 1).replace(".png", "_rec.png")
#         mask = img.replace(data_class, "ground_truth").replace("ground_truth", data_class, 1).replace(".png",
#                                                                                                       "_mask.png")
#         shutil.copy(img, os.path.join(os.path.dirname(train_imgs[0]), f"{train_img_num + idx}.png"))
#         shutil.copy(rec, os.path.join(
#             os.path.dirname(train_imgs[0]).replace(data_class, "rec").replace("rec", data_class, 1),
#             f"{train_img_num + idx}_rec.png"))
#         shutil.copy(mask, os.path.join(
#             os.path.dirname(train_imgs[0]).replace(data_class, "ground_truth").replace("ground_truth", data_class, 1),
#             f"{train_img_num + idx}_mask.png"))

from templates import *

device = 'cuda:0'
conf = ffhq256_autoenc()
conf.T_eval = 50
# print(conf.name)
model = LitModel(conf)
state = torch.load(f'../checkpoints/{conf.name}/epoch3183-step2383238.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)
model.setup()
model.evaluate_scores()
