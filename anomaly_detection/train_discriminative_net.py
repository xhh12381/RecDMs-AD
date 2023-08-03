import torch

import wandb
from sklearn.metrics import roc_auc_score, average_precision_score

from discriminative_net import DiscriminativeSubNetwork
from focal_loss import FocalLoss
from templates import *
from data_loader import *


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def evaluate(model_seg, c):
    obj_ap_pixel_list = []  # 列表中的每个元素代表一个图片的pixel-wise ap指标
    obj_auroc_pixel_list = []  # pixel-wise auroc指标
    obj_ap_image_list = []  # image-wise ap指标
    obj_auroc_image_list = []  # image-wise auroc指标
    img_dim = 256

    model_seg.eval()

    dataset = MVTecDRAEMTestDataset(f"datasets/{c}/test/{c}", resize_shape=[256, 256])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 预测的训练集中每个像素的异常分数
    total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    mask_cnt = 0

    anomaly_score_gt = []
    anomaly_score_prediction = []

    with tqdm(dataloader, desc="Eval...", leave=False) as data_iterator:
        for i_batch, sample_batched in enumerate(data_iterator):
            gray_batch = sample_batched["image"].cuda()

            true_mask = sample_batched["mask"]
            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))  # (b, c, h, w) -> (h, w, c)
            # is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]  # numpy->float
            is_normal = int(true_mask_cv.max())
            anomaly_score_gt.append(is_normal)

            gray_rec = sample_batched["rec"].cuda()

            with torch.no_grad():
                joined_in = torch.cat((gray_rec.detach(), gray_batch.detach()), dim=1)
                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)

            save_dir = f"./outputs/{c}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_image(gray_batch[0], os.path.join(save_dir, f"{i_batch}.png"))
            save_image(true_mask[0], os.path.join(save_dir, f"{i_batch}_mask.png"))
            save_image(gray_rec[0], os.path.join(save_dir, f"{i_batch}_rec.png"))
            save_image(torch.argmax(out_mask_sm, dim=1, keepdim=True)[0].float(),
                       os.path.join(save_dir, f"{i_batch}_seg.png"))
            diff = torch.abs(gray_batch - gray_rec)
            diff = torch.sum(diff, dim=1, keepdim=True)
            diff = (diff - diff.min()) / (diff.max() - diff.min())
            save_image(diff[0], os.path.join(save_dir, f"{i_batch}_diff.png"))

            out_mask_cv = out_mask_sm[0, 1, :, :].detach().cpu().numpy()
            # 经过avg_pool2d后，shape不变
            out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[:, 1:, :, :], 21, stride=1,
                                                               padding=21   // 2).cpu().detach().numpy()
            image_score = np.max(out_mask_averaged)

            anomaly_score_prediction.append(image_score)

            flat_true_mask = true_mask_cv.flatten()
            flat_out_mask = out_mask_cv.flatten()
            total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
            total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
            mask_cnt += 1

        # image-wise指标
        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt)
        auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
        ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

        # pixel-wise指标
        total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
        total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
        total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)

        obj_ap_pixel_list.append(ap_pixel)
        obj_auroc_pixel_list.append(auroc_pixel)
        obj_auroc_image_list.append(auroc)
        obj_ap_image_list.append(ap)
        print("AUC Image:  " + str(auroc))
        print("AP Image:  " + str(ap))
        print("AUC Pixel:  " + str(auroc_pixel))
        print("AP Pixel:  " + str(ap_pixel))
        model_seg.train()
        return auroc, ap, auroc_pixel, ap_pixel


def main(c):
    os.environ["WANDB_API_KEY"] = "c8ddccb46bd291d16653fdaf18a8de222c8ed9af"
    os.environ["WANDB_MODE"] = "dryrun"
    experiment = wandb.init(project='diffae', resume='allow', anonymous='must')
    device = 'cuda:0'
    # conf = ffhq256_autoenc()
    # # print(conf.name)
    # model = LitModel(conf)
    # state = torch.load(f'../checkpoints/{conf.name}/epoch3183-step2383238.ckpt', map_location='cpu')
    # model.load_state_dict(state['state_dict'], strict=False)
    # model.ema_model.eval()
    # model.ema_model.to(device)

    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    model_seg.train()
    model_seg.to(device)
    model_seg.apply(weights_init)
    optimizer = torch.optim.Adam(model_seg.parameters(), lr=1e-4)
    # l1 = torch.nn.L1Loss()
    fl = FocalLoss()

    # dataset = MVTecDRAEMTrainDataset("../datasets/hazelnut/train/good/", "../datasets/dtd/images/",
    #                                  # resize_shape=[256, 256])
    dataset = MVTecDRAEMTestDataset(f"datasets/{c}/train/{c}", resize_shape=[256, 256])
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)  # bs:32
    xT = torch.randn(size=(1, 3, 256, 256))
    image_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # data = next(iter(dataloader))

    for epoch in range(1, 50 + 1):
        epoch_l1_loss = 0
        epoch_focal_loss = 0
        epoch_loss = 0
        with tqdm(dataloader, desc=f"{c} DiscriminativeNetwork training...epoch:{epoch}/50",
                  leave=False) as data_iterator:
            for i, data in enumerate(data_iterator):
                # image = data["image"]
                # anomaly_mask = data["anomaly_mask"].to(device)
                # augmented_image = data["augmented_image"].to(device)
                # with torch.no_grad():
                #     cond = model.encode(image_transform(augmented_image))
                #     x_T = xT.repeat((len(cond), 1, 1, 1)).to(device)
                #     pred = model.render(x_T, cond, T=100)
                #
                # save_image(image[0], f"test_hazelnut/{i}.png")
                # save_image(augmented_image[0], f"test_hazelnut/{i}_aug.png")
                # save_image(anomaly_mask[0], f"test_hazelnut/{i}_mask.png")
                # save_image(pred[0], f"test_hazelnut/{i}_rec.png")

                rec = data['rec'].to(device)
                image = data['image'].to(device)
                anomaly_mask = data['mask'].to(device)
                joined_in = torch.cat((rec, image), dim=1)
                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)
                # l1_loss = l1(out_mask_sm[:, 1, :].unsqueeze(dim=1), anomaly_mask)
                focal_loss = fl(out_mask_sm, anomaly_mask)
                # loss = l1_loss + focal_loss
                loss = focal_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # epoch_l1_loss += l1_loss.item()
                # epoch_focal_loss += focal_loss.item()
                epoch_loss += loss.item()
                data_iterator.set_postfix(**{'loss (batch)': loss.item()})
        auroc, ap, auroc_pixel, ap_pixel = evaluate(model_seg, c)
        experiment.log({
            # 'l1_loss': epoch_l1_loss,
            # 'focal_loss': epoch_focal_loss,
            'train loss': epoch_loss,
            'auroc': auroc,
            'ap': ap,
            'auroc_pixel': auroc_pixel,
            'ap_pixel': ap_pixel,
            'epoch': epoch
        })
        if not os.path.exists(f'../checkpoints/{c}'):
            os.makedirs(f'../checkpoints/{c}')
        if auroc > 0.7 and auroc_pixel > 0.6:
            torch.save(model_seg.state_dict(), f"../checkpoints/{c}/auroc={auroc}-auroc_pixel={auroc_pixel}.pth")
        if epoch % 10 == 0:
            torch.save(model_seg.state_dict(), f"../checkpoints/{c}/epoch={epoch}.pth")

    # test_dataset = MVTecDataset(source="../datasets/", classname="hazelnut", split="test")
    # test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    # xT = torch.randn(size=(1, 3, 256, 256))
    #
    # scores = []
    # labels_gt = []
    # for epoch in range(1, 100 + 1):
    #     epoch_loss = 0
    #     with tqdm(test_dataloader, desc=f"DiscriminativeSubNetwork training...{epoch}/100", leave=False) as data_iterator:
    #         for data in data_iterator:
    #             labels_gt.extend(data["is_anomaly"].numpy().tolist())
    #             image = data["image"]
    #             mask_gt = data["mask"]
    #             gen = (data["gen"] + 1) / 2
    #             ori = (image + 1) / 2
    #             # with torch.no_grad():
    #             #     cond = model.encode(image.to(device))
    #             #     x_T = xT.repeat((len(cond), 1, 1, 1)).to(device)
    #             #     pred = model.render(x_T, cond, T=250)
    #
    #             joined_in = torch.cat((gen, ori), dim=1).to(device)
    #             out_mask = model_seg(joined_in)
    #             out_mask_sm = torch.softmax(out_mask, dim=1)
    #             loss = loss_fn(out_mask_sm, mask_gt)
    #
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #
    #             epoch_loss += loss.item()
    #             data_iterator.set_postfix(**{'loss (batch)': loss.item()})
    #     experiment.log({
    #         'train loss': epoch_loss,
    #         'epoch': epoch
    #     })
    #     if epoch % 20 == 0:
    #         torch.save(model_seg.state_dict(), f"../checkpoints/epoch={epoch}.pth")


if __name__ == '__main__':
    # class_list = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    #               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    class_list = ['chl']
    for c in class_list:
        # device = "cuda:0"
        # model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        # model_seg.load_state_dict(torch.load(f"../checkpoints/new/{c}.pth"))
        # model_seg.to(device)
        # evaluate(model_seg, c)
        main(c)
