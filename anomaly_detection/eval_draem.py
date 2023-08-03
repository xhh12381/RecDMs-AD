from sklearn.metrics import roc_auc_score, average_precision_score

from data_loader import MVTecDRAEMTestDataset
from discriminative_net import DiscriminativeSubNetwork
from templates import *


def write_results_to_file(run_name, image_auc, pixel_auc, image_ap, pixel_ap):
    if not os.path.exists('./outputs/'):
        os.makedirs('./outputs/')

    fin_str = "img_auc," + run_name
    for i in image_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += "," + str(np.round(np.mean(image_auc), 3))
    fin_str += "\n"
    fin_str += "pixel_auc," + run_name
    for i in pixel_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += "," + str(np.round(np.mean(pixel_auc), 3))
    fin_str += "\n"
    fin_str += "img_ap," + run_name
    for i in image_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += "," + str(np.round(np.mean(image_ap), 3))
    fin_str += "\n"
    fin_str += "pixel_ap," + run_name
    for i in pixel_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += "," + str(np.round(np.mean(pixel_ap), 3))
    fin_str += "\n"
    fin_str += "--------------------------\n"

    with open("./outputs/results.txt", 'a+') as file:
        file.write(fin_str)


def test(obj_names):
    obj_ap_pixel_list = []  # 列表中的每个元素代表一个图片的pixel-wise ap指标
    obj_auroc_pixel_list = []  # pixel-wise auroc指标
    obj_ap_image_list = []  # image-wise ap指标
    obj_auroc_image_list = []  # pixel-wise auroc指标
    run_name = None
    for obj_name in obj_names:
        img_dim = 256
        run_name = obj_name + '_'

        conf = ffhq256_autoenc()
        # print(conf.name)
        model = LitModel(conf)
        state = torch.load(f'../checkpoints/{conf.name}/epoch3183-step2383238.ckpt', map_location='cpu')
        model.load_state_dict(state['state_dict'], strict=False)
        model.ema_model.eval()
        model.ema_model.cuda()

        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.load_state_dict(torch.load(f"../checkpoints/new/{obj_name}.pth"))
        model_seg.eval()
        model_seg.cuda()

        dataset = MVTecDRAEMTestDataset(f"../datasets/mvtec/{obj_name}/test/", resize_shape=[256, 256])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        xT = torch.randn(size=(1, 3, 256, 256))
        image_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        # 预测的训练集中每个像素的异常分数
        total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        mask_cnt = 0

        anomaly_score_gt = []
        anomaly_score_prediction = []

        display_images = torch.zeros((16, 3, 256, 256)).cuda()
        display_gt_images = torch.zeros((16, 3, 256, 256)).cuda()
        display_out_masks = torch.zeros((16, 1, 256, 256)).cuda()
        display_in_masks = torch.zeros((16, 1, 256, 256)).cuda()
        cnt_display = 0
        display_indices = np.random.randint(len(dataloader), size=(16,))  # 在测试集中随机选16个图片进行展示

        with tqdm(dataloader, desc="Sampling...", leave=False) as data_iterator:
            for i_batch, sample_batched in enumerate(data_iterator):

                gray_batch = sample_batched["image"].cuda()

                is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]  # numpy->float
                anomaly_score_gt.append(is_normal)
                true_mask = sample_batched["mask"]
                true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))  # (b, c, h, w) -> (h, w, c)

                with torch.no_grad():
                    cond = model.encode(image_transform(gray_batch))
                    x_T = xT.repeat((len(cond), 1, 1, 1)).cuda()
                    gray_rec = model.render(x_T, cond, T=250)
                    joined_in = torch.cat((gray_rec.detach() * 2 - 1, gray_batch), dim=1)
                    out_mask = model_seg(joined_in)
                    out_mask_sm = torch.softmax(out_mask, dim=1)
                save_dir = f"./outputs/{obj_name}"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_image(gray_batch[0], os.path.join(save_dir, f"{i_batch}.png"))
                save_image(true_mask[0], os.path.join(save_dir, f"{i_batch}_mask.png"))
                save_image(gray_rec[0], os.path.join(save_dir, f"{i_batch}_rec.png"))
                save_image(torch.argmax(out_mask_sm, dim=1, keepdim=True)[0].float(),
                           os.path.join(save_dir, f"{i_batch}_seg.png"))

                if i_batch in display_indices:
                    # 等价于torch.argmax(out_mask_sm, dim=1, keepdim=True)
                    # 但是t_mask中的元素都是[0, 1]的小数，表示该像素点为异常的概率，后面会处理成灰度图而不是二值图
                    t_mask = out_mask_sm[:, 1, :, :]
                    display_images[cnt_display] = gray_rec[0]
                    display_gt_images[cnt_display] = gray_batch[0]
                    display_out_masks[cnt_display] = t_mask[0]
                    display_in_masks[cnt_display] = true_mask[0]
                    cnt_display += 1

                out_mask_cv = out_mask_sm[0, 1, :, :].detach().cpu().numpy()
                # 经过avg_pool2d后，shape不变
                out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[:, 1:, :, :], 21, stride=1,
                                                                   padding=21 // 2).cpu().detach().numpy()
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
        print(obj_name)
        print("AUC Image:  " + str(auroc))
        print("AP Image:  " + str(ap))
        print("AUC Pixel:  " + str(auroc_pixel))
        print("AP Pixel:  " + str(ap_pixel))
        print("=====================================")

    print(run_name)
    print("AUC Image mean:  " + str(np.mean(obj_auroc_image_list)))
    print("AP Image mean:  " + str(np.mean(obj_ap_image_list)))
    print("AUC Pixel mean:  " + str(np.mean(obj_auroc_pixel_list)))
    print("AP Pixel mean:  " + str(np.mean(obj_ap_pixel_list)))

    write_results_to_file(run_name, obj_auroc_image_list, obj_auroc_pixel_list, obj_ap_image_list, obj_ap_pixel_list)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, default=0)
    parser.add_argument('--data_path', action='store', type=str,
                        default="D:/PycharmProjects/improved-diffusion-main/datasets/mvtec/")
    parser.add_argument('--checkpoint_path', action='store', type=str,
                        default="./checkpoints/DRAEM_checkpoints/")

    args = parser.parse_args()

    obj_list = ['capsule',
                'bottle',
                'carpet',
                'leather',
                'pill',
                'transistor',
                'tile',
                'cable',
                'zipper',
                'toothbrush',
                'metal_nut',
                'hazelnut',
                'screw',
                'grid',
                'wood'
                ]
    # obj_list = ["hazelnut"]

    with torch.cuda.device(args.gpu_id):
        test(obj_list)
