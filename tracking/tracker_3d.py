## TransCenter: Transformers with Dense Representations for Multiple-Object Tracking
## Copyright Inria
## Year 2022
## Contact : yihong.xu@inria.fr
##
## TransCenter is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## TransCenter is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program, TransCenter.  If not, see <http://www.gnu.org/licenses/> and the LICENSE file.
##
##
## TransCenter has code derived from
## (1) 2020 fundamentalvision.(Apache License 2.0: https://github.com/fundamentalvision/Deformable-DETR)
## (2) 2020 Philipp Bergmann, Tim Meinhardt. (GNU General Public License v3.0 Licence: https://github.com/phil-bergmann/tracking_wo_bnw)
## (3) 2020 Facebook. (Apache License Version 2.0: https://github.com/facebookresearch/detr/)
## (4) 2020 Xingyi Zhou.(MIT License: https://github.com/xingyizhou/CenterTrack)
## (5) 2021 Wenhai Wang. (Apache License Version 2.0: https://github.com/whai362/PVT/blob/v2/LICENSE)
##
## TransCenter uses packages from
## (1) 2019 Charles Shang. (BSD 3-Clause Licence: https://github.com/CharlesShang/DCNv2)
## (2) 2020 fundamentalvision.(Apache License 2.0: https://github.com/fundamentalvision/Deformable-DETR)
import copy
import sys
import os

# dirty insert path #
# cur_path = os.path.realpath(__file__)
# cur_dir = "/".join(cur_path.split('/')[:-2])
# sys.path.insert(0, cur_dir)
import time

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import lap
from post_processing.decode import generic_decode
from tracking.box_merger import merge_bounding_boxes
from util import box_ops


class Tracker:
    """The main tracking file, here is where magic happens."""
    # only track pedestrian
    cl = 1

    def __init__(self, obj_detect, tracker_cfg, postprocessor=None, main_args=None):

        self.obj_detect_T = obj_detect
        self.obj_detect_F = obj_detect
        self.public_detections = tracker_cfg['public_detections']
        self.inactive_patience = tracker_cfg['inactive_patience']
        self.do_reid = tracker_cfg['do_reid']
        self.max_features_num = tracker_cfg['max_features_num']
        self.reid_sim_threshold = tracker_cfg['reid_sim_threshold']
        self.do_align = tracker_cfg['do_align']
        self.motion_model_cfg = tracker_cfg['motion_model']
        self.postprocessor = postprocessor
        self.main_args = main_args

        self.inactive_tracks_T = []
        self.inactive_tracks_F = []
        self.track_num_T = 0
        self.track_num_F = 0
        self.im_index = 0
        self.results_T = {}
        self.results_F = {}
        self.results_3d = {}
        self.img_features = None
        self.encoder_pos_encoding = None
        self.transforms = transforms.ToTensor()
        self.last_image = None
        self.pre_sample_T = None
        self.pre_sample_F = None
        self.sample_T = None
        self.sample_F = None
        self.pre_img_features = None
        self.pre_encoder_pos_encoding = None
        self.flow = None
        self.det_thresh = main_args.track_thresh + 0.1
        self.match_thresh_T = 0.85
        self.match_thresh_F = 0.9
        self.match_3d = None

    def reset(self, hard=True):
        self.tracks_T = []
        self.tracks_F = []
        self.tracks_3d = []
        self.inactive_tracks_T = []
        self.inactive_tracks_F = []
        self.last_image_T = None
        self.last_image_F = None
        self.pre_sample_T = None
        self.pre_sample_F = None
        self.obj_detect_T.pre_memory = None
        self.obj_detect_F.pre_memory = None
        self.sample_T = None
        self.sample_F = None
        self.pre_img_features_T = None
        self.pre_img_features_F = None
        self.pre_encoder_pos_encoding = None
        self.flow = None
        self.obj_detect_T.masks_flatten = None
        self.obj_detect_F.masks_flatten = None
        self.match_3d = None
        if hard:
            self.track_num_F = 0
            self.track_num_T = 0
            self.results_T = {}
            self.results_F = {}
            self.results_3d = {}
            self.im_index = 0

    def linear_assignment(self, cost_matrix, thresh):
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        matches, unmatched_a, unmatched_b = [], [], []

        cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)

        for ix, mx in enumerate(x):
            if mx >= 0:
                matches.append([ix, mx])
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
        matches = np.asarray(matches)

        # matches = [[match_row_idx, match_column_idx]...], it gives you all the matches (assignments)
        # unmatched_a gives all the unmatched row indexes
        # unmatched_b gives all the unmatched column indexes
        return matches, unmatched_a, unmatched_b

    def tracks_to_inactive(self, tracks):
        self.tracks = [t for t in self.tracks if t not in tracks]
        for t in tracks:
            t.pos = t.last_pos[-1]
        self.inactive_tracks += tracks

    def add(self, new_det_pos, new_det_scores, new_det_features, view='top'):
        """Initializes new Track objects and saves them."""
        num_new = new_det_pos.size(0)
        if view == 'top':
            for i in range(num_new):
                self.tracks_T.append(Track(
                    new_det_pos[i].view(1, -1),
                    new_det_scores[i],
                    self.track_num_T + i,
                    new_det_features[i].view(1, -1),
                    self.inactive_patience,
                    self.max_features_num,
                    self.motion_model_cfg['n_steps'] if self.motion_model_cfg['n_steps'] > 0 else 1
                ))
            self.track_num_T += num_new
        else:
            for i in range(num_new):
                self.tracks_F.append(Track(
                    new_det_pos[i].view(1, -1),
                    new_det_scores[i],
                    self.track_num_F + i,
                    new_det_features[i].view(1, -1),
                    self.inactive_patience,
                    self.max_features_num,
                    self.motion_model_cfg['n_steps'] if self.motion_model_cfg['n_steps'] > 0 else 1
                ))
            self.track_num_F += num_new
        # self.track_num += num_new

    # need to discriminate T or F
    def tracks_dets_matching_tracking(self, raw_dets, raw_scores, pre2cur_cts, pos=None, reid_cts=None,
                                      reid_feats=None, view='top'):
        """
        raw_dets and raw_scores are clean (only ped class and filtered by a threshold
        """
        if pos is None:
            if view == 'top':
                pos = self.get_pos()[0].clone()
            else:
                pos = self.get_pos()[1].clone()

        # if len(self.tracks_T) != len(self.tracks_F):
        #     self.track_num = np.ceil((len(self.tracks_T) + len(self.tracks_T))/2)

        # iou matching #
        assert pos.nelement() > 0 and pos.shape[0] == pre2cur_cts.shape[0]

        # todo we can directly output warped_pos for faster inference #
        if raw_dets.nelement() > 0:
            assert raw_dets.shape[0] == raw_scores.shape[0]
            pos_w = pos[:, [2]] - pos[:, [0]]
            pos_h = pos[:, [3]] - pos[:, [1]]

            warped_pos = torch.cat([pre2cur_cts[:, [0]] - 0.5 * pos_w,
                                    pre2cur_cts[:, [1]] - 0.5 * pos_h,
                                    pre2cur_cts[:, [0]] + 0.5 * pos_w,
                                    pre2cur_cts[:, [1]] + 0.5 * pos_h], dim=1)

            # index low-score dets #
            inds_low = raw_scores > 0.1
            inds_high = raw_scores < self.main_args.track_thresh
            inds_second = torch.logical_and(inds_low, inds_high)
            dets_second = raw_dets[inds_second]
            scores_second = raw_scores[inds_second]
            reid_cts_second = reid_cts[inds_second]

            # if len(self.tracks_T) > len(self.tracks_F):
            #     self.match_thresh_T = 0.3
            #     self.match_thresh_F = 1
            # else:
            #     self.match_thresh_T = 1
            #     self.match_thresh_F = 0.3

            # index high-score dets #
            remain_inds = raw_scores > self.main_args.track_thresh
            dets = raw_dets[remain_inds]
            scores_keep = raw_scores[remain_inds]
            reid_cts_keep = reid_cts[remain_inds]

            # Step 1: first assignment #
            if len(dets) > 0:
                assert dets.shape[0] == scores_keep.shape[0]
                # matching with gIOU
                iou_dist = box_ops.generalized_box_iou(pos, dets)

                # todo fuse with dets scores here.
                if self.main_args.fuse_scores:
                    iou_dist *= scores_keep[None, :]

                iou_dist = 1 - iou_dist

                # todo recover inactive tracks here ?
                if view == 'top':
                    matches, u_track, u_detection = self.linear_assignment(iou_dist.cpu().numpy(), thresh=0.85)
                else:
                    matches, u_track, u_detection = self.linear_assignment(iou_dist.cpu().numpy(), thresh=0.9)

                det_feats = F.grid_sample(reid_feats, reid_cts_keep.unsqueeze(0).unsqueeze(0),
                                          mode='bilinear', padding_mode='zeros', align_corners=False)[:, :, 0, :]
                if matches.shape[0] > 0:
                    # update track dets, scores #
                    for idx_track, idx_det in zip(matches[:, 0], matches[:, 1]):
                        if view == 'top':
                            t = self.tracks_T[idx_track]
                        else:
                            t = self.tracks_F[idx_track]
                        t.pos = dets[[idx_det]]
                        t.add_features(det_feats[:, :, idx_det])
                        t.score = scores_keep[[idx_det]]
                pos_birth = dets[u_detection, :]
                scores_birth = scores_keep[u_detection]
                dets_features_birth = det_feats[0, :, u_detection].transpose(0, 1)
            else:
                # no detection, kill all
                if view == 'top':
                    u_track = list(range(len(self.tracks_T)))
                else:
                    u_track = list(range(len(self.tracks_F)))
                pos_birth = torch.zeros(size=(0, 4), device=pos.device, dtype=pos.dtype)
                scores_birth = torch.zeros(size=(0,), device=pos.device).long()
                dets_features_birth = torch.zeros(size=(0, 64), device=pos.device, dtype=pos.dtype)
            # Step 2: second assignment #
            # get remained tracks
            if len(u_track) > 0:
                if len(dets_second) > 0:
                    remained_tracks_pos = pos[u_track]
                    track_indices = copy.deepcopy(u_track)
                    # matching with gIOU
                    iou_dist = 1 - box_ops.generalized_box_iou(remained_tracks_pos, dets_second)  # [0, 2]

                    matches, u_track_second, u_detection_second = self.linear_assignment(iou_dist.cpu().numpy(),
                                                                                         thresh=0.4)  # stricter with low-score dets
                    # update u_track here
                    u_track = [track_indices[t_idx] for t_idx in u_track_second]

                    if matches.shape[0] > 0:
                        second_det_feats = F.grid_sample(reid_feats,
                                                         reid_cts_second[matches[:, 1]].unsqueeze(0).unsqueeze(0),
                                                         mode='bilinear', padding_mode='zeros', align_corners=False)[:,
                                           :, 0, :]
                        # update track dets, scores #
                        for cc, (idx_match, idx_det) in enumerate(zip(matches[:, 0], matches[:, 1])):
                            idx_track = track_indices[idx_match]
                            if view == 'top':
                                t = self.tracks_T[idx_track]
                            else:
                                t = self.tracks_F[idx_track]
                            t.pos = dets_second[[idx_det]]
                            gather_feat_t = second_det_feats[:, :, cc]
                            t.add_features(gather_feat_t)
                            t.score = scores_second[[idx_det]]
        else:
            # no detection, kill all
            if view == 'top':
                num = len(self.tracks_T)
            else:
                num = len(self.tracks_F)
            u_track = list(range(num))
            pos_birth = torch.zeros(size=(0, 4), device=pos.device, dtype=pos.dtype)
            scores_birth = torch.zeros(size=(0,), device=pos.device).long()
            dets_features_birth = torch.zeros(size=(0, 64), device=pos.device, dtype=pos.dtype)

        # put inactive tracks for top view
        self.new_tracks = []
        if view == 'top':
            for i, t in enumerate(self.tracks_T):
                if i in u_track:  # inactive
                    t.pos = t.last_pos[-1]
                    self.inactive_tracks_T += [t]
                else:  # keep
                    self.new_tracks.append(t)
            self.tracks_T = self.new_tracks
        else:
            # put inactive tracks for front view
            for i, t in enumerate(self.tracks_F):
                if i in u_track:  # inactive
                    t.pos = t.last_pos[-1]
                    self.inactive_tracks_F += [t]
                else:  # keep
                    self.new_tracks.append(t)
            self.tracks_F = self.new_tracks

        return [pos_birth, scores_birth, dets_features_birth]

    def detect_tracking_duel_vit(self, batch):

        [ratio, padw, padh] = batch['trans']
        pos_T = self.get_pos()[0].clone()
        pos_F = self.get_pos()[1].clone()

        no_pre_cts_T = False
        no_pre_cts_F = False

        if pos_T.shape[0] > 0:
            # make pre_cts #
            # bboxes to centers
            hm_h, hm_w = self.sample_T.tensors.shape[2], self.sample_T.tensors.shape[3]
            bboxes = pos_T.clone()

            # bboxes
            bboxes[:, 0] += bboxes[:, 2]
            bboxes[:, 1] += bboxes[:, 3]
            pre_cts_T = bboxes[:, 0:2] / 2.0

            # to input image plane
            pre_cts_T *= ratio
            pre_cts_T[:, 0] += padw
            pre_cts_T[:, 1] += padh
            pre_cts_T[:, 0] = torch.clamp(pre_cts_T[:, 0], 0, hm_w - 1)
            pre_cts_T[:, 1] = torch.clamp(pre_cts_T[:, 1], 0, hm_h - 1)
            # to output image plane
            pre_cts_T /= self.main_args.down_ratio
        else:
            pre_cts_T = torch.zeros(size=(2, 2), device=pos_T.device, dtype=pos_T.dtype)
            no_pre_cts_T = True
            print("No Pre Cts in top view!")

        if pos_F.shape[0] > 0:
            # make pre_cts #
            # bboxes to centers
            hm_h, hm_w = self.sample_F.tensors.shape[2], self.sample_F.tensors.shape[3]
            bboxes = pos_F.clone()

            # bboxes
            bboxes[:, 0] += bboxes[:, 2]
            bboxes[:, 1] += bboxes[:, 3]
            pre_cts_F = bboxes[:, 0:2] / 2.0

            # to input image plane
            pre_cts_F *= ratio
            pre_cts_F[:, 0] += padw
            pre_cts_F[:, 1] += padh
            pre_cts_F[:, 0] = torch.clamp(pre_cts_F[:, 0], 0, hm_w - 1)
            pre_cts_F[:, 1] = torch.clamp(pre_cts_F[:, 1], 0, hm_h - 1)
            # to output image plane
            pre_cts_F /= self.main_args.down_ratio
        else:
            pre_cts_F = torch.zeros(size=(2, 2), device=pos_F.device, dtype=pos_F.dtype)
            no_pre_cts_F = True
            print("No Pre Cts in front view!")

        outputs_T = self.obj_detect_T(samples=self.sample_T, pre_samples=self.pre_sample_T,
                                      pre_cts=pre_cts_T.clone().unsqueeze(0))
        outputs_F = self.obj_detect_F(samples=self.sample_F, pre_samples=self.pre_sample_F,
                                      pre_cts=pre_cts_F.clone().unsqueeze(0))

        # # post processing #
        output_T = {k: v[-1] for k, v in outputs_T.items() if k != 'boxes'}
        output_F = {k: v[-1] for k, v in outputs_F.items() if k != 'boxes'}

        # 'hm' is not _sigmoid!
        output_T['hm'] = torch.clamp(output_T['hm'].sigmoid(), min=1e-4, max=1 - 1e-4)
        output_F['hm'] = torch.clamp(output_F['hm'].sigmoid(), min=1e-4, max=1 - 1e-4)

        decoded_T = generic_decode(output_T, K=self.main_args.K, opt=self.main_args)
        decoded_F = generic_decode(output_F, K=self.main_args.K, opt=self.main_args)

        out_scores_T = decoded_T['scores'][0]
        out_scores_F = decoded_F['scores'][0]
        labels_out_F = decoded_F['clses'][0].int() + 1
        labels_out_T = decoded_T['clses'][0].int() + 1

        # reid features #
        # torch.Size([1, 64, 152, 272])

        if no_pre_cts_T:
            pre2cur_cts_T = torch.zeros_like(pos_T)[..., :2]
        else:
            pre2cur_cts_T = self.main_args.down_ratio * (decoded_T['tracking'][0] + pre_cts_T)
            pre2cur_cts_T[:, 0] -= padw
            pre2cur_cts_T[:, 1] -= padh
            pre2cur_cts_T /= ratio
        if no_pre_cts_F:
            pre2cur_cts_F = torch.zeros_like(pos_F)[..., :2]
        else:
            pre2cur_cts_F = self.main_args.down_ratio * (decoded_F['tracking'][0] + pre_cts_F)
            pre2cur_cts_F[:, 0] -= padw
            pre2cur_cts_F[:, 1] -= padh
            pre2cur_cts_F /= ratio

        # extract reid features #
        boxes_T = decoded_T['bboxes'][0].clone()
        boxes_F = decoded_F['bboxes'][0].clone()
        reid_cts_T = torch.stack([0.5 * (boxes_T[:, 0] + boxes_T[:, 2]), 0.5 * (boxes_T[:, 1] + boxes_T[:, 3])], dim=1)
        reid_cts_F = torch.stack([0.5 * (boxes_F[:, 0] + boxes_F[:, 2]), 0.5 * (boxes_F[:, 1] + boxes_F[:, 3])], dim=1)
        reid_cts_T[:, 0] /= outputs_T['reid'][0].shape[3]
        reid_cts_T[:, 0] /= outputs_F['reid'][0].shape[2]
        reid_cts_F[:, 1] /= outputs_T['reid'][0].shape[3]
        reid_cts_F[:, 1] /= outputs_F['reid'][0].shape[2]
        reid_cts_T = torch.clamp(reid_cts_T, min=0.0, max=1.0)
        reid_cts_F = torch.clamp(reid_cts_F, min=0.0, max=1.0)
        reid_cts_T = (2.0 * reid_cts_T - 1.0)
        reid_cts_F = (2.0 * reid_cts_F - 1.0)
        # print(reid_cts.shape)

        out_boxes_T = decoded_T['bboxes'][0] * self.main_args.down_ratio
        out_boxes_F = decoded_F['bboxes'][0] * self.main_args.down_ratio
        out_boxes_T[:, 0::2] -= padw
        out_boxes_F[:, 0::2] -= padw
        out_boxes_T[:, 1::2] -= padh
        out_boxes_F[:, 1::2] -= padh
        out_boxes_T /= ratio
        out_boxes_F /= ratio

        # filtered by scores #
        # out_boxes_T, out_scores_T, labels_out_T, reid_cts_T = merge_bounding_boxes(out_boxes_T, out_scores_T,
        #                                                                            labels_out_T, reid_cts_T)
        # out_boxes_F, out_scores_F, labels_out_F, reid_cts_F = merge_bounding_boxes(out_boxes_F, out_scores_F,
        #                                                                            labels_out_F, reid_cts_F)
        filtered_idx_T = labels_out_T == 1
        filtered_idx_F = labels_out_F == 2  # todo warning, wrong for multiple classes
        out_scores_T = out_scores_T[filtered_idx_T]
        out_scores_F = out_scores_F[filtered_idx_F]
        out_boxes_T = out_boxes_T[filtered_idx_T]
        out_boxes_F = out_boxes_F[filtered_idx_F]
        reid_cts_T = reid_cts_T[filtered_idx_T]
        reid_cts_F = reid_cts_F[filtered_idx_F]

        filtered_outer_F = out_boxes_F[:, 0] < 2400
        out_scores_F = out_scores_F[filtered_outer_F]
        out_boxes_F = out_boxes_F[filtered_outer_F]
        reid_cts_F = reid_cts_F[filtered_outer_F]

        filtered_outer_T = out_boxes_T[:, 0] < 2500
        out_scores_T = out_scores_T[filtered_outer_T]
        out_boxes_T = out_boxes_T[filtered_outer_T]
        reid_cts_T = reid_cts_T[filtered_outer_T]

        if self.main_args.clip:  # for mot20 clip box
            _, _, orig_h, orig_w = batch['img'].shape
            out_boxes_T[:, 0::2] = torch.clamp(out_boxes_T[:, 0::2], 0, orig_w - 1)
            out_boxes_T[:, 1::2] = torch.clamp(out_boxes_T[:, 1::2], 0, orig_h - 1)
            out_boxes_F[:, 0::2] = torch.clamp(out_boxes_F[:, 0::2], 0, orig_w - 1)
            out_boxes_F[:, 1::2] = torch.clamp(out_boxes_F[:, 1::2], 0, orig_h - 1)

        # post processing #
        return out_boxes_T, out_scores_T, pre2cur_cts_T, pos_T, reid_cts_T, outputs_T['reid'][0], \
               out_boxes_F, out_scores_F, pre2cur_cts_F, pos_F, reid_cts_F, outputs_F['reid'][0]

    def get_pos(self):
        """Get the positions of all active tracks."""
        if len(self.tracks_T) == 1:
            pos_T = self.tracks_T[0].pos
        elif len(self.tracks_T) > 1:
            pos_T = torch.cat([t.pos for t in self.tracks_T], dim=0)
        else:
            pos_T = torch.zeros(size=(0, 4), device=self.sample_T.tensors.device).float()
        if len(self.tracks_F) == 1:
            pos_F = self.tracks_F[0].pos
        elif len(self.tracks_F) > 1:
            pos_F = torch.cat([t.pos for t in self.tracks_F], dim=0)
        else:
            pos_F = torch.zeros(size=(0, 4), device=self.sample_F.tensors.device).float()
        return pos_T, pos_F

    # did not use in this file, did not modify
    def get_features(self):
        """Get the features of all active tracks."""
        if len(self.tracks) == 1:
            features = self.tracks[0].features
        elif len(self.tracks) > 1:
            features = torch.cat([t.features for t in self.tracks], 0)
        else:
            features = torch.zeros(size=(0,), device=self.sample.tensors.device).float()
        return features

    # did not use in this file, did not modify
    def get_inactive_features(self):
        """Get the features of all inactive tracks."""
        if len(self.inactive_tracks) == 1:
            features = self.inactive_tracks[0].features
        elif len(self.inactive_tracks) > 1:
            features = torch.cat([t.features for t in self.inactive_tracks], 0)
        else:
            features = torch.zeros(0).cuda()
        return features

    def reid(self, blob, new_det_pos, new_det_scores, new_det_features, view='top'):
        """Tries to ReID inactive tracks with provided detections."""

        if self.do_reid:
            # calculate appearance distances
            dist_mat, pos = [], []
            if view == 'top' and len(self.inactive_tracks_T) > 0:
                for t in self.inactive_tracks_T:
                    dist_mat.append(torch.cat([t.test_features(feat.view(1, -1))
                                               for feat in new_det_features], dim=1))
                    pos.append(t.pos)
                if len(dist_mat) > 1:
                    dist_mat = torch.cat(dist_mat, 0)
                    pos = torch.cat(pos, 0)
                else:
                    dist_mat = dist_mat[0]
                    pos = pos[0]
                if self.main_args.iou_recover_T:
                    iou_dist = 1 - box_ops.generalized_box_iou(pos, new_det_pos)
                    matches, u_track, u_detection = self.linear_assignment(iou_dist.cpu().numpy(), thresh=0.85)
                else:
                    matches, u_track, u_detection = self.linear_assignment(dist_mat.cpu().numpy(),
                                                                           thresh=self.reid_sim_threshold)
                assigned = []
                remove_inactive = []
                if matches.shape[0] > 0:
                    for r, c in zip(matches[:, 0], matches[:, 1]):
                        # inactive tracks reactivation #
                        if dist_mat[r, c] <= self.reid_sim_threshold or not self.main_args.iou_recover_T:
                            t = self.inactive_tracks_T[r]
                            self.tracks_T.append(t)
                            t.add_features(new_det_features[c].view(1, -1))
                            t.count_inactive = 0
                            t.pos = new_det_pos[c].view(1, -1)
                            t.reset_last_pos()
                            assigned.append(c)
                            remove_inactive.append(t)
                for t in remove_inactive:
                    self.inactive_tracks_T.remove(t)

                keep = [i for i in range(new_det_pos.size(0)) if i not in assigned]
                if len(keep) > 0:
                    new_det_pos = new_det_pos[keep]
                    new_det_scores = new_det_scores[keep]
                    new_det_features = new_det_features[keep]
                else:
                    new_det_pos = torch.zeros(size=(0, 4), device=self.sample_T.tensors.device).float()
                    new_det_scores = torch.zeros(size=(0,), device=self.sample_T.tensors.device).long()
                    new_det_features = torch.zeros(size=(0, 128), device=self.sample_T.tensors.device).float()

            if view == 'front' and len(self.inactive_tracks_F) > 0:
                for t in self.inactive_tracks_F:
                    dist_mat.append(torch.cat([t.test_features(feat.view(1, -1))
                                               for feat in new_det_features], dim=1))
                    pos.append(t.pos)
                if len(dist_mat) > 1:
                    dist_mat = torch.cat(dist_mat, 0)
                    pos = torch.cat(pos, 0)
                else:
                    dist_mat = dist_mat[0]
                    pos = pos[0]
                if self.main_args.iou_recover_F:
                    iou_dist = 1 - box_ops.generalized_box_iou(pos, new_det_pos)
                    matches, u_track, u_detection = self.linear_assignment(iou_dist.cpu().numpy(), thresh=0.9)
                else:
                    matches, u_track, u_detection = self.linear_assignment(dist_mat.cpu().numpy(),
                                                                           thresh=self.reid_sim_threshold)
                assigned = []
                remove_inactive = []
                if matches.shape[0] > 0:
                    for r, c in zip(matches[:, 0], matches[:, 1]):
                        # inactive tracks reactivation #
                        if dist_mat[r, c] <= self.reid_sim_threshold or not self.main_args.iou_recover_F:
                            t = self.inactive_tracks_F[r]
                            self.tracks_F.append(t)
                            t.add_features(new_det_features[c].view(1, -1))
                            t.count_inactive = 0
                            t.pos = new_det_pos[c].view(1, -1)
                            t.reset_last_pos()
                            assigned.append(c)
                            remove_inactive.append(t)
                for t in remove_inactive:
                    self.inactive_tracks_F.remove(t)
                # if len(u_track) > 0 and len(self.tracks_T) > (len(self.tracks_F) + len(u_track)):
                #     for t in self.inactive_tracks_F:
                #         self.tracks_F.append(t)
                #     self.inactive_tracks_F.clear()
                keep = [i for i in range(new_det_pos.size(0)) if i not in assigned]
                if len(keep) > 0:
                    new_det_pos = new_det_pos[keep]
                    new_det_scores = new_det_scores[keep]
                    new_det_features = new_det_features[keep]
                else:
                    new_det_pos = torch.zeros(size=(0, 4), device=self.sample_F.tensors.device).float()
                    new_det_scores = torch.zeros(size=(0,), device=self.sample_F.tensors.device).long()
                    new_det_features = torch.zeros(size=(0, 128), device=self.sample_F.tensors.device).float()

        return new_det_pos, new_det_scores, new_det_features

    def add_features(self, new_features):
        """Adds new appearance features to active tracks."""
        for t, f in zip(self.tracks, new_features):
            t.add_features(f.view(1, -1))

    @torch.no_grad()
    def step_reidV3_pre_tracking_vit(self, blob):
        """This function should be called every time step to perform tracking with a blob
        containing the image information.
        """
        # Nested tensor #
        self.sample_T = blob['samples_T']
        self.sample_F = blob['samples_F']

        if self.pre_sample_T is None:
            self.pre_sample_T = self.sample_T
        if self.pre_sample_F is None:
            self.pre_sample_F = self.sample_F

        ###########################
        # Look for new detections #
        ###########################
        # detect
        det_pos_T, det_scores_T, pre2cur_cts_T, mypos_T, reid_cts_T, reid_features_T, \
        det_pos_F, det_scores_F, pre2cur_cts_F, mypos_F, reid_cts_F, reid_features_F = self.detect_tracking_duel_vit(
            blob)
        ##################
        # Predict tracks #
        ##################
        if len(self.tracks_T):
            [det_pos_T, det_scores_T, dets_features_birth_T] = self.tracks_dets_matching_tracking(raw_dets=det_pos_T,
                                                                                                  raw_scores=det_scores_T,
                                                                                                  pre2cur_cts=pre2cur_cts_T,
                                                                                                  pos=mypos_T,
                                                                                                  reid_cts=reid_cts_T,
                                                                                                  reid_feats=reid_features_T,
                                                                                                  view='top')
        else:
            dets_features_birth_T = F.grid_sample(reid_features_T, reid_cts_T.unsqueeze(0).unsqueeze(0),
                                                  mode='bilinear', padding_mode='zeros',
                                                  align_corners=False)[:, :, 0, :].transpose(1, 2)[0]

        if len(self.tracks_F):
            [det_pos_F, det_scores_F, dets_features_birth_F] = self.tracks_dets_matching_tracking(
                raw_dets=det_pos_F, raw_scores=det_scores_F,
                pre2cur_cts=pre2cur_cts_F, pos=mypos_F,
                reid_cts=reid_cts_F, reid_feats=reid_features_F, view='front')
        else:
            dets_features_birth_F = F.grid_sample(reid_features_F, reid_cts_F.unsqueeze(0).unsqueeze(0),
                                                  mode='bilinear', padding_mode='zeros',
                                                  align_corners=False)[:, :, 0, :].transpose(1, 2)[0]

        #####################
        # Create new tracks #
        #####################

        valid_dets_idx_T = det_scores_T >= self.det_thresh
        valid_dets_idx_F = det_scores_F >= self.det_thresh
        det_pos_T = det_pos_T[valid_dets_idx_T]
        det_pos_F = det_pos_F[valid_dets_idx_F]
        det_scores_T = det_scores_T[valid_dets_idx_T]
        det_scores_F = det_scores_F[valid_dets_idx_F]
        dets_features_birth_T = dets_features_birth_T[valid_dets_idx_T]
        dets_features_birth_F = dets_features_birth_F[valid_dets_idx_F]

        if det_pos_T.nelement() > 0:

            assert det_pos_T.shape[0] == dets_features_birth_T.shape[0] == det_scores_T.shape[0]
            # try to re-identify tracks
            det_pos_T, det_scores_T, dets_features_birth_T = self.reid(blob, det_pos_T, det_scores_T,
                                                                       dets_features_birth_T, view='top')
            assert det_pos_T.shape[0] == dets_features_birth_T.shape[0] == det_scores_T.shape[0]

            # add new
            if det_pos_T.nelement() > 0:
                self.add(det_pos_T, det_scores_T, dets_features_birth_T, view='top')

        if det_pos_F.nelement() > 0:

            assert det_pos_F.shape[0] == dets_features_birth_F.shape[0] == det_scores_F.shape[0]
            # try to re-identify tracks
            det_pos_F, det_scores_F, dets_features_birth_F = self.reid(blob, det_pos_F, det_scores_F,
                                                                       dets_features_birth_F, view='front')
            assert det_pos_F.shape[0] == dets_features_birth_F.shape[0] == det_scores_F.shape[0]

            # add new
            if det_pos_F.nelement() > 0:
                self.add(det_pos_F, det_scores_F, dets_features_birth_F, view='front')

        ####################
        # Generate Results #
        ####################
        for t in self.tracks_T:
            if t.id not in self.results_T.keys():
                self.results_T[t.id] = {}
            self.results_T[t.id][self.im_index] = np.concatenate([t.pos[0].cpu().numpy(), [t.score.cpu()]])
        for t in self.tracks_F:
            if t.id not in self.results_F.keys():
                self.results_F[t.id] = {}
            self.results_F[t.id][self.im_index] = np.concatenate([t.pos[0].cpu().numpy(), [t.score.cpu()]])

        new_inactive_tracks = []
        for t in self.inactive_tracks_T:
            t.count_inactive += 1
            if t.has_positive_area() and t.count_inactive <= self.inactive_patience:
                new_inactive_tracks.append(t)
        self.inactive_tracks_T = new_inactive_tracks

        new_inactive_tracks = []
        for t in self.inactive_tracks_F:
            t.count_inactive += 1
            if t.has_positive_area() and t.count_inactive <= self.inactive_patience:
                new_inactive_tracks.append(t)
        self.inactive_tracks_F = new_inactive_tracks
        ######################
        # Generate 3d Tracks #
        ######################
        self.tracks_3d = self.match_3d.get_3d_tracks(self.tracks_T, self.tracks_F, self.tracks_3d)  # list of dict
        for t in self.tracks_3d:
            if t['id'] not in self.results_3d.keys():
                self.results_3d[t['id']] = {}
            self.results_3d[t['id']][self.im_index] = t
        self.im_index += 1

    def get_results(self):
        return self.results_T, self.results_F, self.results_3d


class Track(object):
    """This class contains all necessary for every individual track."""

    def __init__(self, pos, score, track_id, features, inactive_patience, max_features_num, mm_steps):
        self.id = track_id
        self.pos = pos
        self.score = score
        self.features = deque([features])
        self.ims = deque([])
        self.count_inactive = 0
        self.inactive_patience = inactive_patience
        self.max_features_num = max_features_num
        self.last_pos = deque([pos.clone()], maxlen=mm_steps + 1)
        self.last_v = torch.Tensor([])
        self.gt_id = None

    def has_positive_area(self):
        return self.pos[0, 2] > self.pos[0, 0] and self.pos[0, 3] > self.pos[0, 1]

    def add_features(self, features):
        """Adds new appearance features to the object."""
        self.features.append(features)
        if len(self.features) > self.max_features_num:
            self.features.popleft()

    def test_features(self, test_features):
        """Compares test_features to features of this Track object"""
        if len(self.features) > 1:
            features = torch.cat(list(self.features), dim=0)
        else:
            features = self.features[0]
        features = features.mean(0, keepdim=True)
        dist = F.pairwise_distance(features, test_features, keepdim=True)
        return dist

    def reset_last_pos(self):
        self.last_pos.clear()
        self.last_pos.append(self.pos.clone())
