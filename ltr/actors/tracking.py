from . import BaseActor
import torch
import numpy as np

class HCATActor(BaseActor):
    """ Actor for training the HCAT"""
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'search_images', 'template_images', 'search_anno'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        import time
        outputs = self.net(data['search_images'], data['template_images'])

        # generate labels
        targets =[]
        targets_origin = data['search_anno']
        _, _, h, w = data['search_images'].shape
        targets_origin[:, 0] += targets_origin[:, 2] / 2
        targets_origin[:, 0] /= w
        targets_origin[:, 1] += targets_origin[:, 3] / 2
        targets_origin[:, 1] /= h
        targets_origin[:, 2] /= w
        targets_origin[:, 3] /= h
        targets_origin = targets_origin.unsqueeze(1)
        for i in range(len(targets_origin)):
            target_origin = targets_origin[i]
            target = {}
            target['boxes'] = target_origin
            label = np.array([0])
            label = torch.tensor(label, device=data['search_anno'].device)
            target['labels'] = label
            targets.append(target)

        # Compute loss
        # outputs:(center_x, center_y, width, height)
        loss_dict = self.objective(outputs, targets)
        weight_dict = self.objective.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Return training stats
        stats = {'Loss/total': losses.item(),
                 'Loss/ce': loss_dict['loss_ce'].item(),
                 'Loss/bbox': loss_dict['loss_bbox'].item(),
                 'Loss/giou': loss_dict['loss_giou'].item(),
                 'iou': loss_dict['iou'].item()
                 }

        return losses, stats
