from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import numpy as np
import math
import torchvision.transforms.functional as tvisf
import cv2
import torch
import torch.nn.functional as F
import time
'''Refine module & Pytracking base trackers'''
# from common_path import *
import os
'''2020.4.24 Use new pytracking library(New DiMP)'''
from pytracking.evaluation import Tracker
'''2020.4.15 ARcm_seg model'''
'''other utils'''
from pytracking.vot20_utils import *
from pysot_toolkit.trackers.net_wrappers import NetWithBackbone
import pytracking.evaluation.vot2020 as vot
''''''
class TRANST(object):

    def __init__(self, name, net, feature_size, window_penalty=0.49, penalty_k=0, exemplar_size=128, instance_size=256):
        self.name = name
        self.net = net
        self.window_penalty = window_penalty
        self.penalty_k = penalty_k
        self.exemplar_size = exemplar_size
        self.instance_size = instance_size
        self.feature_size = feature_size


    def _convert_score(self, score):

        score = score.permute(2, 1, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 0].cpu().numpy()
        return score

    def _convert_bbox(self, delta):

        delta = delta.permute(2, 1, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        return delta

    def _convert_mask(self, delta):

        delta = delta.squeeze(0).squeeze(0)
        delta = delta.data.cpu().numpy()

        return delta

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: rgb based image
            pos: center position
            model_sz: exemplar size
            original_sz: original size
            avg_chans: channel average
        """
        # tic = time.time()
        resize_factor = original_sz / model_sz

        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1

        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        r, c, k = im.shape
        im_context = im[max(0,int(context_ymin)):min(int(context_ymax)+1,r),max(0,int(context_xmin)):min(int(context_xmax)+1,c),:]
        r_context, c_context, _ = im_context.shape
        r_context_resize = round(r_context / resize_factor)
        c_context_resize = round(c_context / resize_factor)
        left_pad = round(left_pad / resize_factor)
        top_pad = round(top_pad / resize_factor)
        right_pad = round(right_pad / resize_factor)
        bottom_pad = round(bottom_pad / resize_factor)

        im_patch_context = cv2.resize(im_context, (c_context_resize, r_context_resize))

        te_im = np.zeros([model_sz,model_sz,k])
        te_im[top_pad:top_pad + r_context_resize, left_pad:left_pad + c_context_resize, :] = im_patch_context

        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c_context_resize, :] = avg_chans
        if bottom_pad:
            te_im[r_context_resize + top_pad:, left_pad:left_pad + c_context_resize, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c_context_resize + left_pad:, :] = avg_chans
        # print(time.time()-tic)


        # if not np.array_equal(model_sz, original_sz):
        #     im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = te_im
        _,r,c = im_patch.shape
        if not r == c == model_sz:
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))


        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)


        im_patch = im_patch.cuda()

        # 0.3ms
        return im_patch

    def map_mask_back(self, im, center_pos, instance_size, s_x, mask, mode=cv2.BORDER_REPLICATE):
        """ Extracts a crop centered at target_bb box, of size search_area_factor times target_bb(Both height and width)

        args:
            im - cv image
            target_bb - target box [x, y, w, h]
            search_area_factor - Ratio of crop size to target size
            output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

        returns:
            cv image - extracted crop
            float - the factor by which the crop has been resized to make the crop size equal output_size
        """
        H, W = (im.shape[0], im.shape[1])
        base = np.zeros((H, W))
        x_center, y_center = center_pos.tolist()

        # Crop image

        if s_x < 1 or s_x < 1:
            raise Exception('Too small bounding box.')
        c = (s_x + 1) / 2

        x1 = int(np.floor(x_center - c + 0.5))
        x2 = int(x1 + s_x - 1)

        y1 = int(np.floor(y_center - c + 0.5))
        y2 = int(y1 + s_x -1)

        x1_pad = int(max(0., -x1))
        y1_pad = int(max(0., -y1))
        x2_pad = int(max(0., x2 - W + 1))
        y2_pad = int(max(0., y2 - H + 1))

        '''pad base'''
        base_padded = cv2.copyMakeBorder(base, y1_pad, y2_pad, x1_pad, x2_pad, mode)
        '''Resize mask'''
        mask_rsz = cv2.resize(mask, (s_x, s_x))
        '''fill region with mask'''
        base_padded[y1 + y1_pad:y2 + y1_pad + 1, x1 + x1_pad:x2 + x1_pad + 1] = mask_rsz.copy()
        '''crop base_padded to get final mask'''
        final_mask = base_padded[y1_pad:y1_pad + H, x1_pad:x1_pad + W]
        assert (final_mask.shape == (H, W))
        return final_mask

    def constraint_mask(self, mask, bbox):
        """
        mask: shape (H, W)
        bbox: list [x1, y1, w, h]
        """
        x1 = np.int(np.floor(bbox[0]))
        y1 = np.int(np.floor(bbox[1]))
        x2 = np.int(np.ceil(bbox[0]+bbox[2]))
        y2 = np.int(np.ceil(bbox[1]+bbox[3]))
        mask[0:y1+1,:] = 0
        mask[y2:,:] = 0
        mask[:,0:x1+1] = 0
        mask[:,x2:] = 0
        if mask.max() == 0:
            yp1 = np.int(np.floor(bbox[1]+bbox[3]/4))
            yp2 = np.int(np.ceil(bbox[1]+3*bbox[3]/4))
            xp1 = np.int(np.floor(bbox[0]+bbox[2]/4))
            xp2 = np.int(np.ceil(bbox[0]+3*bbox[2]/4))
            mask[yp1:yp2,xp1:xp2] = 1
        return mask

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.net.initialize()
        self.features_initialized = True

    def initialize(self, image, box):
        hanning = np.hanning(self.feature_size)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()
        # Initialize
        self.initialize_features()

        region = box
        gt_bbox_np = np.array(region).astype(np.float32)
        bbox = gt_bbox_np

        self.center_pos = np.array([bbox[0] + bbox[2] / 2,
                                    bbox[1] + bbox[3] / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # while(True):
        #     cv2.imshow('image', image)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break


        # calculate z crop size
        w_z = self.size[0] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_z = self.size[1] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_z = math.ceil(math.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(image, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(image, self.center_pos,
                                    self.exemplar_size,
                                    s_z, self.channel_average)

        # normalize
        # normalize
        self.mean_ = torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).cuda()
        self.std_ = torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).cuda()
        z_crop /= 255
        z_crop -= self.mean_
        z_crop /= self.std_

        # initialize template feature
        self.net.template(z_crop, self.feature_size)

    def track(self, image):
        # calculate x crop size
        w_x = self.size[0] + (4 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_x = self.size[1] + (4 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_x = math.ceil(math.sqrt(w_x * h_x))

        # get crop
        x_crop_ori = self.get_subwindow(image, self.center_pos,
                                        self.instance_size,
                                        round(s_x), self.channel_average)

        # normalize
        x_crop = x_crop_ori
        x_crop /= 255
        x_crop -= self.mean_
        x_crop /= self.std_

        # track
        outputs = self.net.track(x_crop)
        score = self._convert_score(outputs['pred_logits'])
        pred_bbox = self._convert_bbox(outputs['pred_boxes'])

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]/s_x, self.size[1]/s_x)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * self.penalty_k)
        pscore = penalty * score
        # pscore = score
        # window penalty
        pscore = pscore * (1 - self.window_penalty) + \
                 self.window * self.window_penalty

        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx]
        bbox = bbox * s_x
        cx = bbox[0] + self.center_pos[0] - s_x / 2
        cy = bbox[1] + self.center_pos[1] - s_x / 2
        width = bbox[2]
        height = bbox[3]

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, image.shape[:2])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        # update state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])


        x_crop_return = x_crop_ori.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        # out = {'target_bbox': bbox,
        #        'target_mask': final_mask,
        #        'best_score': pscore[best_idx]}
        return bbox, x_crop_return

def _convert_anno_to_list(vot_anno):
    vot_anno = [vot_anno[0], vot_anno[1], vot_anno[2], vot_anno[3]]
    return vot_anno

def run_vot_exp(name, window, penalty_k, net_path, save_root, VIS=False):
    torch.set_num_threads(1)
    # torch.cuda.set_device(CUDA_ID)  # set GPU id
    if VIS and (not os.path.exists(save_root)):
        os.mkdir(save_root)

    net = NetWithBackbone(net_path=net_path, use_gpu=True)
    tracker = TRANST(name=name, net=net, feature_size=16, window_penalty=window, penalty_k=penalty_k, exemplar_size=128, instance_size=256)

    handle = vot.VOT("rectangle")
    selection = handle.region()
    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)
    if VIS:
        '''for vis'''
        seq_name = imagefile.split('/')[-3]
        save_v_dir = os.path.join(save_root,seq_name)
        if not os.path.exists(save_v_dir):
            os.mkdir(save_v_dir)
        cur_time = int(time.time() % 10000)
        save_dir = os.path.join(save_v_dir, str(cur_time))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB) # Right
    # mask given by the toolkit ends with the target (zero-padding to the right and down is needed)
    # mask = make_full_size(selection, (image.shape[1], image.shape[0]))
    bbox = _convert_anno_to_list(selection)
    tracker.initialize(image, bbox)

    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
        b1, search = tracker.track(image)
        handle.report(vot.Rectangle(*b1))
        if VIS:
            '''Visualization'''
            # original image
            image_ori = image[:,:,::-1].copy() # RGB --> BGR
            image_name = imagefile.split('/')[-1]
            save_path = os.path.join(save_dir, image_name)
            cv2.imwrite(save_path, image_ori)
            # dimp box
            image_b = image_ori.copy()
            cv2.rectangle(image_b, (int(b1[0]), int(b1[1])),
                          (int(b1[0] + b1[2]), int(b1[1] + b1[3])), (0, 0, 255), 2)
            image_b_name = image_name.replace('.jpg','_bbox.jpg')
            save_path = os.path.join(save_dir, image_b_name)
            cv2.imwrite(save_path, image_b)
            # search region
            search_bgr = search[:,:,::-1].copy()
            search_name = image_name.replace('.jpg', '_search.jpg')
            save_path = os.path.join(save_dir, search_name)
            cv2.imwrite(save_path, search_bgr)
            # search region mask
            # search_bgr_m = search_bgr.astype(np.float32)
            # search_bgr_m[:, :, 1] += 127.0 * search_m
            # search_bgr_m[:, :, 2] += 127.0 * search_m
            # contours, _ = cv2.findContours(search_m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # search_bgr_m = cv2.drawContours(search_bgr_m, contours, -1, (0, 255, 255), 4)
            # search_bgr_m = search_bgr_m.clip(0,255).astype(np.uint8)
            # search_name_m = image_name.replace('.jpg', '_search_mask.jpg')
            # save_path = os.path.join(save_dir, search_name_m)
            # cv2.imwrite(save_path, search_bgr_m)
            # original image + mask
            # image_m = image_ori.copy().astype(np.float32)
            # image_m[:, :, 1] += 127.0 * m
            # image_m[:, :, 2] += 127.0 * m
            # contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # image_m = cv2.drawContours(image_m, contours, -1, (0, 255, 255), 2)
            # image_m = image_m.clip(0, 255).astype(np.uint8)
            # image_mask_name_m = image_name.replace('.jpg', '_mask.jpg')
            # save_path = os.path.join(save_dir, image_mask_name_m)
            # cv2.imwrite(save_path, image_m)
