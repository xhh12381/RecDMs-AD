import os.path

import torch

from anomaly_detection.data_loader import MVTecDRAEMTestDataset, MVTecDRAEMTrainDataset
from templates import *

device = 'cuda:0'
conf = ffhq256_autoenc()
# print(conf.name)
model = LitModel(conf)
state = torch.load(f'../checkpoints/{conf.name}/epoch=43607-step=1816594.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device)

# class_list = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
#               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
class_list = ['chl']
for c in class_list:
    # dataset = MVTecDRAEMTestDataset(f"../datasets/chl/{c}/test", resize_shape=[256, 256])
    dataset = MVTecDRAEMTrainDataset(f"../datasets/chl/{c}/train/good/", "../datasets/dtd/images/",
                                     resize_shape=[256, 256])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    image_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    x_T = torch.randn(size=(1, 3, 256, 256)).to(device)
    torch.manual_seed(44)

    with tqdm(dataloader, desc=f"{c} Sampling...", leave=False) as data_iterator:
        for i, data in enumerate(data_iterator):
            image = data["image"]
            # image = data["augmented_image"]
            mask = data["mask"]
            with torch.no_grad():
                cond = model.encode(image_transform(image).to(device))
                rec = model.render(x_T, cond, T=250)
            image_save_dir = os.path.join('datasets', c, 'test', c, 'images')
            if not os.path.exists(image_save_dir):
                os.makedirs(image_save_dir)
            save_image(image, os.path.join(image_save_dir, f'{i}.png'))
            rec_save_dir = os.path.join('datasets', c, 'test', 'rec', 'images')
            if not os.path.exists(rec_save_dir):
                os.makedirs(rec_save_dir)
            save_image(rec, os.path.join(rec_save_dir, f"{i}_rec.png"))
            mask_save_dir = os.path.join('datasets', c, 'test', 'ground_truth', 'images')
            if not os.path.exists(mask_save_dir):
                os.makedirs(mask_save_dir)
            save_image(mask, os.path.join(mask_save_dir, f"{i}_mask.png"))
