import logging
import torch
import torch.nn as nn
from dataloader import full_path_loader, CDDloader, full_test_loader
from models.FC_EF_2018 import UNet
from losses import DiceLoss, DiceBCELoss, IoULoss, FocalLoss
logging.basicConfig(level=logging.INFO)
        
def get_loaders(opt):

    logging.info('STARTING Dataset Creation')

    train_full_load, val_full_load = full_path_loader(opt['dataset_dir'])


    train_dataset = CDDloader(train_full_load, aug=opt['augmentation'])
    val_dataset = CDDloader(val_full_load, aug=False)

    logging.info('STARTING Dataloading')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt['batch_size'],
                                               shuffle=True,
                                               num_workers=opt['num_workers'])
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=opt['batch_size'],
                                             shuffle=False,
                                             num_workers=opt['num_workers'])
    return train_loader, val_loader

def get_test_loaders(opt, batch_size=None):

    if not batch_size:
        batch_size = opt['batch_size']

    logging.info('STARTING Dataset Creation')

    test_full_load = full_test_loader(opt['dataset_dir'])

    test_dataset = CDDloader(test_full_load, aug=False)

    logging.info('STARTING Dataloading')


    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=opt['num_workers'])
    return test_loader


def load_model(opt, device):
    """Load the model
    Parameters
    ----------
    opt : dict
        User specified flags/options
    device : string
        device on which to train model
    """
    # device_ids = list(range(opt['num_gpus']))
    model = UNet(opt['in_channel'], opt['out_channel'], use_dropout=False).to(device)

    return model

def get_criterion(opt):
    """get the user selected loss function
    Parameters
    ----------
    opt : dict
        Dictionary of options/flags
    Returns
    -------
    method
        loss function
    """
    if opt['loss_function'] == 'DiceLoss':
        criterion = DiceLoss()
    if opt['loss_function'] == 'DiceBCELoss':
        criterion = DiceBCELoss()
    if opt['loss_function'] == 'FocalLoss':
        criterion = FocalLoss
    if opt['loss_function'] == 'IoULoss':
        criterion = IoULoss()

    return criterion




def initialize_metrics():
    """Generates a dictionary of metrics with metrics as keys
       and empty lists as values
    Returns
    -------
    dict
        a dictionary of metrics
    """
    metrics = {
        'cd_losses': [],
        'cd_corrects': [],
        'cd_precisions': [],
        'cd_recalls': [],
        'cd_f1scores': [],
        'learning_rate': [],
    }

    return metrics

def get_mean_metrics(metric_dict):
    """takes a dictionary of lists for metrics and returns dict of mean values
    Parameters
    ----------
    metric_dict : dict
        A dictionary of metrics
    Returns
    -------
    dict
        dict of floats that reflect mean metric value
    """
    return {k: np.mean(v) for k, v in metric_dict.items()}


def set_metrics(metric_dict, cd_loss, cd_corrects, cd_report, lr):
    """Updates metric dict with batch metrics
    Parameters
    ----------
    metric_dict : dict
        dict of metrics
    cd_loss : dict(?)
        loss value
    cd_corrects : dict(?)
        number of correct results (to generate accuracy
    cd_report : list
        precision, recall, f1 values
    Returns
    -------
    dict
        dict of  updated metrics
    """
    metric_dict['cd_losses'].append(cd_loss.item())
    metric_dict['cd_corrects'].append(cd_corrects.item())
    metric_dict['cd_precisions'].append(cd_report[0])
    metric_dict['cd_recalls'].append(cd_report[1])
    metric_dict['cd_f1scores'].append(cd_report[2])
    metric_dict['learning_rate'].append(lr)

    return metric_dict

def set_test_metrics(metric_dict, cd_corrects, cd_report):

    metric_dict['cd_corrects'].append(cd_corrects.item())
    metric_dict['cd_precisions'].append(cd_report[0])
    metric_dict['cd_recalls'].append(cd_report[1])
    metric_dict['cd_f1scores'].append(cd_report[2])

    return metric_dict
