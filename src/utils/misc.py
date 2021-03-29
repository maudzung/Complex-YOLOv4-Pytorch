import os
import torch
import time

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # or os.makedirs(folder_name, exist_ok=True)


def dump_predicted_labels(file_path, predictions):
    """ Dumps the output labels (KITTI format) on the provided path """
    with open(file_path, 'w') as out_label_file:
        for predicted_box_parameters in predictions:
            out_label = []
            out_label.append(predicted_box_parameters.type) # BB class
            out_label.append(predicted_box_parameters.truncation) # BB truncation
            out_label.append(predicted_box_parameters.occlusion) # BB occlusion level
            out_label.append(predicted_box_parameters.alpha) # BB observation angle
            out_label.append(predicted_box_parameters.box2d[0]) # 2D BB top
            out_label.append(predicted_box_parameters.box2d[1]) # 2D BB left
            out_label.append(predicted_box_parameters.box2d[2]) # 2D BB bottom
            out_label.append(predicted_box_parameters.box2d[3]) # 2D BB right
            out_label.append(predicted_box_parameters.h) # 3D BB height
            out_label.append(predicted_box_parameters.w) # 3D BB width
            out_label.append(predicted_box_parameters.l) # 3D BB length
            out_label.append(predicted_box_parameters.t[0]) # 3D BB x
            out_label.append(predicted_box_parameters.t[1]) # 3D BB y
            out_label.append(predicted_box_parameters.t[2]) # 3D BB z 
            out_label.append(predicted_box_parameters.ry) # BB rotation around y-axis
            out_label.append(predicted_box_parameters.score) # BB prediction score
            
            for box_param in out_label:
                out_label_file.write(str(box_param)+' ')
            out_label_file.write('\n')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def get_message(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()