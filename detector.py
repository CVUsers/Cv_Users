import os
import cv2
import numpy as np
import time
# from utils.nms_wrapper import nms
from configs.CC import Config
import argparse
from layers.functions import Detect, PriorBox
from peleenet import build_net
from data import BaseTransform, VOC_CLASSES
from utils.core import *
import torch

num = 0
def _to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127


parser = argparse.ArgumentParser(description='Pelee Testing')
parser.add_argument('-c', '--config', default='./configs/Pelee_VOC.py')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument('-m', '--trained_model', default='./Pelee_VOC.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('-t', '--thresh', default=0.3, type=float,
                    help='visidutation threshold')
parser.add_argument('--show', default=True,
                    help='Whether to display the images')
args = parser.parse_args()
cfg = Config.fromfile(args.config)

num_classes = cfg.model.num_classes
base = int(np.ceil(pow(num_classes, 1. / 3)))
colors = [_to_color(x, base)
          for x in range(num_classes)]
cats = [_.strip().split(',')[-1]
        for _ in open('./data/coco_labels.txt', 'r').readlines()]
label_config = {'VOC': VOC_CLASSES, 'COCO': tuple(['__background__'] + cats)}
labels = label_config[args.dataset]
ch_labels = ['', '飞机', '自行车', '鸟', '船', '瓶子', '公交车',

             '小汽车', '猫', '椅子', '奶牛', '餐桌', '狗', '马',

             '摩托车', '人', '盆栽', '绵羊', '沙发',

             '火车', '显示屏']


def draw_detection(im, bboxes, scores, cls_inds, fps, thr=0.2):
    global num
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    infos = []
    for i, box in enumerate(bboxes):
        if scores[i] < thr:
            continue
        print(scores[i])

        cls_indx = int(cls_inds[i])
        print(cls_indx)
        # if cls_indx == 7 or cls_indx == 6:
        if cls_indx >=0:
            num += 1
            box = [int(_) for _ in box]
            thick = int((h + w) / 300)
            cv2.rectangle(imgcv,
                          (box[0], box[1]), (box[2], box[3]),
                          colors[cls_indx], thick)
            mess = '%s' % (labels[cls_indx])
            infos.append(ch_labels[cls_indx]+' '+str(scores[i]))
            cv2.putText(imgcv, mess, (box[0], box[1] - 7),
                        0, 1e-3 * h, colors[cls_indx], thick // 3)
            if fps >= 0:
                cv2.putText(imgcv, '%.2f' % fps + ' fps', (w - 160, h - 15),
                            0, 2e-3 * h, (255, 255, 255), thick // 2)

    return infos, imgcv,num


class Pelee_Det(object):

    def __init__(self):

        self.anchor_config = anchors(cfg.model)
        self.priorbox = PriorBox(self.anchor_config)
        self.net = build_net('test', cfg.model.input_size, cfg.model)
        init_net(self.net, cfg, args.trained_model)
        self.net.eval()

        self.num_classes = cfg.model.num_classes

        with torch.no_grad():
            self.priors = self.priorbox.forward()
            # self.net = self.net.cuda()
            # self.priors = self.priors.cuda()
            cudnn.benchmark = True
        self._preprocess = BaseTransform(
            cfg.model.input_size, cfg.model.rgb_means, (2, 0, 1))
        self.detector = Detect(num_classes,
                               cfg.loss.bkg_label, self.anchor_config)

    def detect(self, image):

        loop_start = time.time()
        w, h = image.shape[1], image.shape[0]
        img = self._preprocess(image).unsqueeze(0)
        # if cfg.test_cfg.cuda:
        #     img = img.cuda()
        scale = torch.Tensor([w, h, w, h])
        out = self.net(img)
        boxes, scores = self.detector.forward(out, self.priors)
        boxes = (boxes[0] * scale).cpu().numpy()
        scores = scores[0].cpu().numpy()
        allboxes = []
        count = 0
        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > cfg.test_cfg.score_threshold)[0]
            if len(inds) == 0:
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            soft_nms = cfg.test_cfg.soft_nms
            keep = nms(c_dets, cfg.test_cfg.iou, force_cpu=soft_nms)
            keep = keep[:cfg.test_cfg.keep_per_class]
            c_dets = c_dets[keep, :]
            allboxes.extend([_.tolist() + [j] for _ in c_dets])

        loop_time = time.time() - loop_start
        allboxes = np.array(allboxes)
        boxes = allboxes[:, :4]
        scores = allboxes[:, 4]
        cls_inds = allboxes[:, 5]
        print('all', allboxes[:9], 'boxes', boxes, 'score', scores, 'inds', cls_inds)
        infos, im2show, num= draw_detection(image, boxes, scores,
                                 cls_inds, -1, 0.13)   # todo threshold need to be modify
        return infos, im2show, num


if __name__ == '__main__':

    det = Pelee_Det()
    im = cv2.imread('./test_imgs/demo.png')  # todo threshold = 0.8
    im = cv2.imread('./test_imgs/demo1.png')  # todo threshold = 0
    im = cv2.imread('./test_imgs/c.jpg') # todo threshold = 0.15
    # im = cv2.imread('./test_imgs/a.png')  # todo threshold = 0.4
    # im = cv2.imread('./test_imgs/k.png')  # todo threshold = 0.01
    im = cv2.imread('./test_imgs/h.jpg')  # todo threshold = 0.07
    # im = cv2.imread('./test_imgs/b.jpg')  # todo threshold = 0.15
    im = cv2.imread('./test_imgs/test2.png') # todo threshold = 0.15

    _, im, num = det.detect(im)
    print('there are', str(num), 'objects')
    num = str(num) + 'objects'
    cv2.putText(im, num, (50, 150), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 4)
    cv2.imshow(num, im)

    cv2.waitKey(0)

    cv2.destroyAllWindows()
