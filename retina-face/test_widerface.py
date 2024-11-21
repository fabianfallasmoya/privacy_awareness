from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from sklearn.cluster import DBSCAN
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer


parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset_folder', default='./data/widerface/val/images/', type=str, help='dataset path')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def calculate_bbox_size(detections):
        # Calculate the width and height of each box
        width = torch.abs(detections[:, 2] - detections[:, 0])  # |x2 - x1|
        height = torch.abs(detections[:, 3] - detections[:, 1])  # |y2 - y1|
        # Calculate the normlized area for 640x640 pixels images
        area = width * height / 409600
        return area

def sigmoid(x, threshold=0.1, alpha=10):
    return 1 / (1 + torch.exp(-alpha * (x - threshold)))

def get_landmark_score(detections, alpha=0.1):
    """
    Clusters landmarks within a bounding box if they are too close to each other
    based on an alpha distance proportional to the size of the bounding box.

    Parameters:
    detections: tensor with predictions
    alpha : float
        Proportion of the bounding box size to use as the clustering distance.

    Returns:
    tensor
        Tensor with the landmark scores base on how many clusters were detected.
    """
    # Calculate width and height of the bounding box
    bbox_width = torch.abs(detections[:, 2] - detections[:, 0])  # |x2 - x1|
    bbox_height = torch.abs(detections[:, 3] - detections[:, 1])  # |y2 - y1|

    # Calculate the clustering distance as alpha times the average bbox size
    clustering_distance = alpha * (bbox_width + bbox_height) / 2

    # Convert points to numpy array for DBSCAN
    points_array = np.array(detections[:, 5:15])

    results = torch.zeros(clustering_distance.shape[0])

    for i in range(results.size(0)):
        # Apply DBSCAN clustering with calculated epsilon
        dbscan = DBSCAN(eps=clustering_distance[i].tolist(), min_samples=1)
        labels = dbscan.fit_predict(points_array[i].reshape(-1,2))

        # Count the number of unique clusters
        num_clusters = len(set(labels))

        match num_clusters:
            case 1:
                results[i] = 0
            case 2:
                results[i] = 0.1
            case 3:
                results[i] = 0.25
            case 4:
                results[i] = 0.5
            case 5:
                results[i] = 1
            case _:
                results[i] = 0
    return results

def calculate_pa_sigmoid(detections):
    """
    Calculate Privacy Awareness using conf score, bbox size and landmarks num

    Params:
        detections (x1,y1,x2,y2,cs,lm1x,lm1y,lm2x,lm2y,lm3x,lm3y,lm4x,lm4y,lm5x,lm5y)
    Returns:
        array_with_pa (x1,y1,x2,y2,cs,lm1x,lm1y,lm2x,lm2y,lm3x,lm3y,lm4x,lm4y,lm5x,lm5y, pa, pa_discret)
    """

    # Convert detections to a PyTorch tensor
    detections = torch.tensor(detections)

    # first calculate the bbox size
    area = calculate_bbox_size(detections)

    # then define sigmoid parameters
    alpha = 25
    threshold = 0.1

    # Calculate the PA using the weights formula
    pa = 0.3 * detections[:, 4] + 0.4 * sigmoid(area, threshold=threshold, alpha=alpha) + 0.3 * get_landmark_score(detections)
    # Add the Privacy Awareness as an extra column
    array_with_pa = torch.cat((detections, pa.unsqueeze(1)), dim=1)

    # now classify continuous PA value into PA interval i.e. 0.73 -> 0.8
    pa_intervals = torch.ceil(pa / 0.2) * 0.2

    array_with_pa = torch.cat((array_with_pa, pa_intervals.unsqueeze(1)), dim=1)

    return array_with_pa.numpy()

def exponentialBoostTransform(x, threshold=0.1, alpha=10):
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

def calculate_pa_expBoost(detections):
        """
        Calculate Privacy Awareness using conf score, bbox size and landmarks num

        Params:
            detections (x1,y1,x2,y2,cs,lm1x,lm1y,lm2x,lm2y,lm3x,lm3y,lm4x,lm4y,lm5x,lm5y)
        Returns:
            array_with_pa (x1,y1,x2,y2,cs,lm1x,lm1y,lm2x,lm2y,lm3x,lm3y,lm4x,lm4y,lm5x,lm5y, pa, pa_discret)
        """
        # Convert detections to a PyTorch tensor
        detections = torch.tensor(detections)

        # first calculate the bbox size
        area = calculate_bbox_size(detections)

        # then define sigmoid parameters
        alpha = 30
        threshold = 0.0001

        # Calculate the PA using the weights formula
        pa = 0.3 * detections[:, 4] + 0.4 * exponentialBoostTransform(area, threshold=threshold, alpha=alpha) + 0.3 * get_landmark_score(detections)
        # Add the Privacy Awareness as an extra column
        array_with_pa = torch.cat((detections, pa.unsqueeze(1)), dim=1)

        # now classify continuous PA value into PA interval i.e. 0.73 -> 0.8
        pa_intervals = torch.ceil(pa / 0.2) * 0.2

        array_with_pa = torch.cat((array_with_pa, pa_intervals.unsqueeze(1)), dim=1)

        return array_with_pa.numpy()

if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # testing dataset
    testset_folder = args.dataset_folder
    testset_list = args.dataset_folder[:-7] + "wider_val.txt"

    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    # testing begin
    for i, img_name in enumerate(test_dataset):
        image_path = testset_folder + img_name
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        # testing scale
        target_size = 1600
        max_size = 2150
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if args.origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        _t['forward_pass'].tic()
        loc, conf, landms = net(img)  # forward pass
        _t['forward_pass'].toc()
        _t['misc'].tic()
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        _t['misc'].toc()

        # Calculate PA
        # dets_pa = calculate_pa_sigmoid(dets)
        dets_pa = calculate_pa_expBoost(dets)

        # --------------------------------------------------------------------
        save_name = args.save_folder + img_name[:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        with open(save_name, "w") as fd:
            bboxs = dets_pa
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            for box in bboxs:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                lm1x = int(box[5])
                lm1y = int(box[6])
                lm2x = int(box[7])
                lm2y = int(box[8])
                lm3x = int(box[9])
                lm3y = int(box[10])
                lm4x = int(box[11])
                lm4y = int(box[12])
                lm5x = int(box[13])
                lm5y = int(box[14])
                pa_d = str(box[15])
                pa = str(box[16])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " " + pa + " " + pa_d + " " + str(lm1x) + " " + str(lm1y) + " " + str(lm2x) + " " + str(lm2y) + " " + str(lm3x) + " " + str(lm3y) + " " + str(lm4x) + " " + str(lm4y) + " " + str(lm5x) + " " + str(lm5y) + " \n"
                fd.write(line)

        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))

        # save image
        if args.save_image:
            for b in dets_pa:
                if b[4] < args.vis_thres:
                    continue
                #text = "{:.4f}".format(b[4])
                text = str(b[16])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image
            if not os.path.exists("./results/"):
                os.makedirs("./results/")
            name = "./results/" + str(i) + ".jpg"
            cv2.imwrite(name, img_raw)

