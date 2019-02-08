import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from model.rpn.union_box_layer_tf import union_box_layer
from model.rpn.edge_box_layer_tf import edge_box_layer
from model.rpn.edge_box_layer_tf import edge_whole_layer

from model.structure.structure_inference_spmm import  structure_inference_spmm
from model.structure.structure import  _Structure_inference
from model.structure.external_feature import  _External

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align_whole = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()
        self.Structure_inference = _Structure_inference(2048,2048,2048)
        # self.External = _External(2048)
    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        # ========= Union Box ==========
        whole_box = union_box_layer(rois, im_info)
        whole_box = whole_box.reshape(whole_box.shape[0], 1, 5)
        whole = torch.from_numpy(whole_box)
        whole = whole.type(torch.cuda.FloatTensor)
        # whole = whole_box.view([-1, 5])

        # edges = edge_box_layer(rois, im_info)
        # edges = torch.from_numpy(edges)
        # edge = edges.view([-1, 12])

        edges_all = edge_whole_layer(rois, im_info)
        edges_all = torch.from_numpy(edges_all)

        # whole_rois = torch.cat((whole, rois), 1)

        rois = Variable(rois)


        # print rois.size()
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
            whole_pool_feat = self.RCNN_roi_align_whole(base_feat, whole.view(-1,5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))
            whole_pool_feat = self.RCNN_roi_pool(base_feat, whole.view(-1,5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)
        whole_pool_feat = self._head_to_tail(whole_pool_feat)

        ##########structure_inference_spmm#################

        # pooled_feat = structure_inference_spmm(pooled_feat , whole_pool_feat, edges, rois.size()[1])
        pooled_feat = self.Structure_inference(edges_all, pooled_feat ,whole_pool_feat,rois.size()[1])

        # print 'pooled_feat.shape:   ',  pooled_feat.shape
        # print 'rois.shape:   ', rois.shape
        # print 'edges.shape: ', edges.shape

        #coordinate = self.coor_fc( rois[:,:,1:].reshape(rois.shape[1], 4) )
        #pooled_feat = torch.cat(( coordinate ,pooled_feat),1)
        #pooled_feat = torch.add(coordinate, pooled_feat)


        # #########  external_dim ###########
        #
        # external_feature = rois[:,:,3:].view([128,2])
        # pooled_feat = self.External(pooled_feat,external_feature)


        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()
        def normal_init1(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight_u.data.normal_(mean, stddev)
                m.weight_ua.data.normal_(mean, stddev)
                m.Concat_w.data.normal_(mean, stddev)
                m.Concat_w2.data.normal_(mean, stddev)
                nn.init.constant(m.weight_confusion, 1)
                # nn.init.xavier_uniform_(m.weight_u)
                # nn.init.xavier_uniform_(m.Concat_w)
                # m.bias.data.zero_()

        # def normal_init2(m, mean, stddev, truncated=False):
        #     """
        #     weight initalizer: truncated normal and random normal.
        #     """
        #     # x is a parameter
        #     if truncated:
        #         m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
        #     else:
        #         m.Linear1.weight.data.normal_(mean, stddev)
        #         m.Linear1.bias.data.zero_()
        #         m.Linear2.weight.data.normal_(mean, stddev)
        #         m.Linear2.bias.data.zero_()
        #         m.Linear3.weight.data.normal_(mean, stddev)
        #         m.Linear3.bias.data.zero_()
        #         # m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        normal_init1(self.Structure_inference, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #normal_init(self.coor_fc, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #normal_init2(self.External, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
