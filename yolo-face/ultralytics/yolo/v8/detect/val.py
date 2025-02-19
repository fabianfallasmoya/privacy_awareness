# Ultralytics YOLO 🚀, GPL-3.0 license

import os
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torchvision.transforms as transforms

from ultralytics.yolo.data import build_dataloader
from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.engine.fewshot import create_model
from ultralytics.yolo.utils import DEFAULT_CFG, colorstr, ops, yaml_load
from ultralytics.yolo.utils.checks import check_file, check_requirements
from ultralytics.yolo.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.yolo.utils.plotting import output_to_target, plot_images
from ultralytics.yolo.utils.torch_utils import de_parallel
from ultralytics import globals


class DetectionValidator(BaseValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, logger=None, args=None):
        super().__init__(dataloader, save_dir, pbar, logger, args)
        self.data_dict = yaml_load(check_file(self.args.data), append_filename=True) if self.args.data else None
        self.is_coco = False
        self.class_map = None
        self.metrics = DetMetrics(save_dir=self.save_dir)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.fewshot_model = create_model()


    def preprocess(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes", "pa"]:
            batch[k] = batch[k].to(self.device)

        nb = len(batch["img"])
        self.lb = [torch.cat([batch["cls"], batch["bboxes"], batch["pa"]], dim=-1)[batch["batch_idx"] == i]
                   for i in range(nb)] if self.args.save_hybrid else []  # for autolabelling

        return batch

    def init_metrics(self, model):
        head = model.model[-1] if self.training else model.model.model[-1]
        val = self.data.get('val', '')  # validation path
        self.is_coco = isinstance(val, str) and val.endswith(f'coco{os.sep}val2017.txt')  # is COCO dataset
        self.class_map = ops.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        self.args.save_json |= self.is_coco and not self.training  # run on final val if training COCO
        self.nc = head.nc
        self.names = model.names
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc)
        self.seen = 0
        self.jdict = []
        self.stats = []

    def get_desc(self):
        return ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'Box(P', "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        labels=self.lb,
                                        multi_label=True,
                                        agnostic=self.args.single_cls,
                                        max_det=self.args.max_det)
        return preds

    def update_metrics(self, preds, batch):
        # Metrics
        for si, pred in enumerate(preds):
            idx = batch["batch_idx"] == si
            cls = batch["cls"][idx]
            bbox = batch["bboxes"][idx]
            pa = batch["pa"][idx]
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
            shape = batch["ori_shape"][si]
            correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
            self.seen += 1

            if npr == 0:
                if nl:
                    self.stats.append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            ops.scale_boxes(batch["img"][si].shape[1:], predn[:, :4], shape,
                            ratio_pad=batch["ratio_pad"][si])  # native-space pred

            # Evaluate
            pred_pa = []
            if nl:
                height, width = batch["img"].shape[2:]
                tbox = ops.xywh2xyxy(bbox) * torch.tensor(
                    (width, height, width, height), device=self.device)  # target boxes
                ops.scale_boxes(batch["img"][si].shape[1:], tbox, shape,
                                ratio_pad=batch["ratio_pad"][si])  # native-space labels
                labelsn = torch.cat((cls, tbox, pa), 1)  # native-space labels
                correct_bboxes = self._process_batch(predn, labelsn)
                correct_bboxes_pa, pred_pa = self._process_batch_pa(predn, labelsn, batch["img"][si])
                # TODO: maybe remove these `self.` arguments as they already are member variable
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)
            # NOTE Change correct_bboxes for correct_bboxes_pa
            #self.stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)
            #self.stats.append((correct_bboxes_pa, pred_pa[:, 6], pred[:, 5], cls.squeeze(-1)))  # (PA, pcls, tcls)
            self.stats.append((correct_bboxes_pa, pred_pa[:, 6], pred_pa[:, 7]*5, pa.squeeze(-1)))  # (PA, predicted PA discreto, label PA discreto)

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            # if self.args.save_txt:
            #    save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')

    def get_stats(self):
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]  # to numpy
        if len(stats) and stats[0].any():
            self.metrics.process(*stats)
        self.nt_per_class = np.bincount(stats[-1].astype(int), minlength=self.nc)  # number of targets per class
        return self.metrics.results_dict

    def print_results(self):
        pf = '%22s' + '%11i' * 2 + '%11.3g' * len(self.metrics.keys)  # print format
        self.logger.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            self.logger.warning(
                f'WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels')

        # Print results per class
        if (self.args.verbose or not self.training) and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                self.logger.info(pf % (self.names[c], self.seen, self.nt_per_class[c], *self.metrics.class_result(i)))

        if self.args.plots:
            self.confusion_matrix.plot(save_dir=self.save_dir, names=list(self.names.values()))

    def _process_batch(self, detections, labels):
        """
        Return correct prediction matrix
        Arguments:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
        iou = box_iou(labels[:, 1:5], detections[:, :4])
        correct = np.zeros((detections.shape[0], self.iouv.shape[0])).astype(bool)
        correct_class = labels[:, 0:1] == detections[:, 5]
        for i in range(len(self.iouv)):
            x = torch.where((iou >= self.iouv[i]) & correct_class)  # IoU > threshold and classes match
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                    1).cpu().numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=detections.device)

    def calculate_bbox_size(self, detections):
        # Calculate the width and height of each box
        width = torch.abs(detections[:, 2] - detections[:, 0])  # |x2 - x1|
        height = torch.abs(detections[:, 3] - detections[:, 1])  # |y2 - y1|
        # Calculate the normlized area for 640x640 pixels images
        area = width * height / 409600
        return area

    def sigmoid(self, x, threshold=0.1, alpha=10):
        return 1 / (1 + torch.exp(-alpha * (x - threshold)))

    def calculate_pa_1_sigmoid(self, detections):

        # first calculate the bbox size
        area = self.calculate_bbox_size(detections)

        # then define sigmoid parameters
        alpha = 40
        threshold = 0.001

        # Calculate the PA using the weights formula
        pa = 0.5 * detections[:, 4] + 0.5 * self.sigmoid(area, threshold=threshold, alpha=alpha)
        # Add the Privacy Awareness as an extra column
        array_with_pa = torch.cat((detections, pa.unsqueeze(1)), dim=1)

        # now classify continuous PA value into PA interval i.e. 0.73 -> 0.8
        pa_intervals = torch.ceil(pa / 0.2) * 0.2

        array_with_pa = torch.cat((array_with_pa, pa_intervals.unsqueeze(1)), dim=1)

        return array_with_pa

    def exponentialBoostTransform(self, x, threshold=0.1, alpha=10):
        # x in the proper range [0, 1]
        x = torch.clip(x, 0, 1)

        # a threshold based transformation
        #
        # 1 - np.exp(-alpha * (x - threshold)) is a decreasing exponential function.
        # when x is equal to the threshold, the function equals 0, and as x increases,
        # the function approaches 1 asymptotically (np.maximum is to stabilize the results).
        return torch.where(x > threshold,
                        torch.maximum(1 - torch.exp(-alpha * (x - threshold)), x),
                        x)

    def calculate_pa_1_expBoost(self, detections):

        # first calculate the bbox size
        area = self.calculate_bbox_size(detections)

        # then define sigmoid parameters
        alpha = 300
        threshold = 0.00005

        # Calculate the PA using the weights formula
        pa = 0.75 * detections[:, 4] + 0.25 * self.exponentialBoostTransform(area, threshold=threshold, alpha=alpha)
        # Add the Privacy Awareness as an extra column
        array_with_pa = torch.cat((detections, pa.unsqueeze(1)), dim=1)

        # now classify continuous PA value into PA interval i.e. 0.73 -> 0.8
        pa_intervals = torch.ceil(pa / 0.2) * 0.2

        array_with_pa = torch.cat((array_with_pa, pa_intervals.unsqueeze(1)), dim=1)

        return array_with_pa

    def calculate_pa_1(self, detections):

        # first calculate the bbox size
        area = self.calculate_bbox_size(detections)

        # Calculate the PA using the weights formula
        pa = 0.75 * detections[:, 4] + 0.25 * area
        # Add the Privacy Awareness as an extra column
        array_with_pa = torch.cat((detections, pa.unsqueeze(1)), dim=1)

        # now classify continuous PA value into PA interval i.e. 0.73 -> 0.8
        pa_intervals = torch.ceil(pa / 0.2) * 0.2

        array_with_pa = torch.cat((array_with_pa, pa_intervals.unsqueeze(1)), dim=1)

        return array_with_pa

    def calculate_pa_0(self, detections):

        # Calculate the PA using the weights formula
        pa = detections[:, 4]
        # Add the Privacy Awareness as an extra column
        array_with_pa = torch.cat((detections, pa.unsqueeze(1)), dim=1)

        # now classify continuous PA value into PA interval i.e. 0.73 -> 0.8
        pa_intervals = torch.ceil(pa / 0.2) * 0.2

        array_with_pa = torch.cat((array_with_pa, pa_intervals.unsqueeze(1)), dim=1)

        return array_with_pa

    def crop_image(self, tensor, detections):
        """
        Crop an RGB image from a given pytorch tensor based on a series of coordinates
        and store the cropped images in a list.

        Args:
            tensor (pytorch tensor): The input Image tensor.
            detections (list of detections): A pytorch tensor with the detections info

        Returns:
            list: A list of cropped images (Pillow Image objects).
        """
        tensor = tensor.cpu()

            # Convert tensor to a PIL Image (ensure the tensor is in [H, W, C] format)
        if tensor.ndimension() == 3 and tensor.shape[0] == 3:  # [C, H, W] format
            tensor = tensor.permute(1, 2, 0)  # Rearrange to [H, W, C]

        # Convert to a NumPy array and then to a PIL Image
        image = Image.fromarray((tensor.numpy() * 255).astype('uint8'))

        cropped_images = []
        for coord in detections:
            cropped_image = image.crop((int(coord[0]), int(coord[1]), int(coord[2]), int(coord[3])))  # Crop the image
            cropped_images.append(cropped_image)

        # for idx, cropped_image in enumerate(cropped_images):
        #     cropped_image.show()  # Opens the image in the default image viewer

        return cropped_images

    def get_few_shot_confidence_score(self, cropped_images):
        # init few shot confidence score
        score = torch.empty(len(cropped_images), dtype=torch.float32)

        # Setup transform to go from PIL image to tensor
        transform = transforms.ToTensor()

        # get the few shot confidence score for each image
        for idx, img in enumerate(cropped_images):
            tensor_image = transform(img)
            _, H, W = tensor_image.shape  # Unpack shape: (C, H, W)
            # Check if either dimension is smaller than 7, since the image can't be smaller than the model kernel 7x7
            if H < 7 or W < 7:
                score[idx] = -100
            else:
                # Convert single image tensor to a batch (batch size = 1)
                tensor_image = tensor_image.unsqueeze(0)  # Shape changes from [C, H, W] -> [1, C, H, W]
                few_shot_score = self.fewshot_model(
                    tensor_image.to("cuda", dtype=torch.float32)
                ).detach()
                score[idx] = few_shot_score.data[0]

        # normalize the confidence score
        score = score.to("cuda")
        min_val, max_val = -28, -16
        # Apply min-max normalization
        normalized = (score - min_val) / (max_val - min_val)

        # Clamp values to the range [0, 1]
        normalized = torch.clamp(normalized, 0, 1)

        return normalized


    def calculate_pa_5_sigmoid(self, detections, img):
        # first calculate the bbox size
        area = self.calculate_bbox_size(detections)

        # second, create the cropped image vector based on the detections and the batch image
        cropped_images = self.crop_image(img, detections)

        # then define sigmoid parameters
        alpha = 40
        threshold = 0.001

        # Calculate the PA using the weights formula
        pa = 0.3 * detections[:, 4] + 0.3 * self.sigmoid(area, threshold=threshold, alpha=alpha) + 0.4 * self.get_few_shot_confidence_score(cropped_images)
        # Add the Privacy Awareness as an extra column
        array_with_pa = torch.cat((detections, pa.unsqueeze(1)), dim=1)

        # now classify continuous PA value into PA interval i.e. 0.73 -> 0.8
        pa_intervals = torch.ceil(pa / 0.2) * 0.2

        array_with_pa = torch.cat((array_with_pa, pa_intervals.unsqueeze(1)), dim=1)

        return array_with_pa

    def normalize_pa_labels(self, labels):
        # Transform PA range from [1,5] to [0,1]
        labels[:, 5] = labels[:, 5] / 5
        return labels

    def _process_batch_pa(self, detections, labels, img):
        """
        Return correct prediction matrix for Privacy Awareness
        Arguments:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 6]), class, x1, y1, x2, y2, PA
        Returns:
            correct_pa (array[N, 10]), for 10 IoU levels
        """
        iou = box_iou(labels[:, 1:5], detections[:, :4])
        correct = np.zeros((detections.shape[0], self.iouv.shape[0])).astype(bool)
        correct_class = labels[:, 0:1] == detections[:, 5]
        correct_pa = None

        # predict the privacy awareness
        # normalize the PA labels since their range is [1,5] but we want them in [0,1] 0.2, 0.4, 0.6, 0.8, 1
        labels_pa = self.normalize_pa_labels(labels)

        match globals.eval_case:
            case 1:

                # Case 1: calculate the PA by combining the confidence score and bbox size, index: x1, y1, x2, y2, conf, class, PA (continuous), PA (intervals)
                #detections_pa = self.calculate_pa_1(detections)
                detections_pa = self.calculate_pa_1_sigmoid(detections)
                #detections_pa = self.calculate_pa_1_expBoost(detections)

                correct_pa = labels_pa[:, 5:6] == detections_pa[:, 7]

            case 5:

                # Case 5: calculate the PA by combining YOLO's confidence score, the bbox size, and the Few Shot model confidence score, index: x1, y1, x2, y2, conf, class, PA (continuous), PA (intervals)
                detections_pa = self.calculate_pa_5_sigmoid(detections, img)

                correct_pa = labels_pa[:, 5:6] == detections_pa[:, 7]

            case _:
                # Case 0: calculate the PA using only confidence score, index: x1, y1, x2, y2, conf, class, PA (continuous), PA (intervals)
                detections_pa = self.calculate_pa_0(detections)

                correct_pa = labels_pa[:, 5:6] == detections_pa[:, 7]

        for i in range(len(self.iouv)):
            x = torch.where((iou >= self.iouv[i]) & correct_class & correct_pa)  # IoU > threshold and classes match and correct pa
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                    1).cpu().numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=detections.device), detections_pa


    def get_dataloader(self, dataset_path, batch_size):
        # TODO: manage splits differently
        # calculate stride - check if model is initialized
        gs = max(int(de_parallel(self.model).stride if self.model else 0), 32)
        return create_dataloader(path=dataset_path,
                                 imgsz=self.args.imgsz,
                                 batch_size=batch_size,
                                 stride=gs,
                                 hyp=vars(self.args),
                                 cache=False,
                                 pad=0.5,
                                 rect=True,
                                 workers=self.args.workers,
                                 prefix=colorstr(f'{self.args.mode}: '),
                                 shuffle=False,
                                 seed=self.args.seed)[0] if self.args.v5loader else \
            build_dataloader(self.args, batch_size, img_path=dataset_path, stride=gs, mode="val")[0]

    def plot_val_samples(self, batch, ni):
        plot_images(batch["img"],
                    batch["batch_idx"],
                    batch["cls"].squeeze(-1),
                    batch["bboxes"],
                    paths=batch["im_file"],
                    fname=self.save_dir / f"val_batch{ni}_labels.jpg",
                    names=self.names)

    def plot_predictions(self, batch, preds, ni):
        plot_images(batch["img"],
                    *output_to_target(preds, max_det=15),
                    paths=batch["im_file"],
                    fname=self.save_dir / f'val_batch{ni}_pred.jpg',
                    names=self.names)  # pred

    def pred_to_json(self, predn, filename):
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append({
                'image_id': image_id,
                'category_id': self.class_map[int(p[5])],
                'bbox': [round(x, 3) for x in b],
                'score': round(p[4], 5)})

    def eval_json(self, stats):
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data['path'] / "annotations/instances_val2017.json"  # annotations
            pred_json = self.save_dir / "predictions.json"  # predictions
            self.logger.info(f'\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...')
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements('pycocotools>=2.0.6')
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                for x in anno_json, pred_json:
                    assert x.is_file(), f"{x} file not found"
                anno = COCO(str(anno_json))  # init annotations api
                pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                eval = COCOeval(anno, pred, 'bbox')
                if self.is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = eval.stats[:2]  # update mAP50-95 and mAP50
            except Exception as e:
                self.logger.warning(f'pycocotools unable to run: {e}')
        return stats


def val(cfg=DEFAULT_CFG):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.data = cfg.data or "coco128.yaml"
    validator = DetectionValidator(args=cfg)
    validator(model=cfg.model)


if __name__ == "__main__":
    val()
