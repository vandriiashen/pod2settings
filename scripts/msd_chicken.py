import numpy as np
from pathlib import Path
from tqdm import tqdm
import imageio
import msd_pytorch as mp
import torch
import matplotlib.pyplot as plt
from skimage import morphology
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/
def soft_dice_score(
    output: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score

class DiceLoss(_Loss):
    def __init__(
        self,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        eps: float = 1e-7,
    ):
        """Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super(DiceLoss, self).__init__()
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            y_pred = y_pred.log_softmax(dim=1).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        y_true = y_true.view(bs, -1)
        y_pred = y_pred.view(bs, num_classes, -1)

        y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
        y_true = y_true.permute(0, 2, 1)  # N, C, H*W

        scores = self.compute_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        return soft_dice_score(output, target, smooth, eps, dims)

def train(dataset_folder, nn_name, i):
    '''Trains MSD on the provided training dataset.
    
    :param train_folder: Training dataset folder
    :type train_folder: :class:`pathlib.PosixPath`
    :param nn_name: Network name
    :type nn_name: :class:`str`
    '''
    c_in = 2
    depth = 30
    width = 2
    dilations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    labels = [0, 1, 2]
    c_out = 1
    batch_size = 3
    epochs = 500
    save_folder = Path('../ckpt/')
    log_path = Path('../log/')

    train_input_glob = dataset_folder / 'train' / 'inp' / '*.tiff'
    train_target_glob = dataset_folder / 'train' / 'tg' / '*.tiff'
    val_input_glob = dataset_folder / 'val' / 'inp' / '*.tiff'
    val_target_glob = dataset_folder / 'val' / 'tg' / '*.tiff'

    print("Load training dataset")
    train_ds = mp.ImageDataset(train_input_glob, train_target_glob, labels=labels)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    val_ds = mp.ImageDataset(val_input_glob, val_target_glob, labels=labels)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False)

    model = mp.MSDSegmentationModel(c_in, train_ds.num_labels, depth, width, dilations=dilations)
    #model.criterion = DiceLoss()
    print(model.criterion)

    print("Start estimating normalization parameters")
    model.set_normalization(train_dl)
    print("Done estimating normalization parameters")
    
    print("Starting training...")
    best_validation_error = np.inf
    best_epoch = -1
    validation_error = 0.0
    
    (save_folder / '{}_{:02d}'.format(nn_name, i)).mkdir(exist_ok = True)
    (log_path / '{}_{:02d}'.format(nn_name, i)).mkdir(exist_ok = True)
    logger = SummaryWriter(log_path / '{}_{:02d}'.format(nn_name, i))

    for epoch in tqdm(range(epochs)):
        model.train(train_dl, 1)
        train_error = model.validate(train_dl)
        print(f"{epoch:05} Training error: {train_error: 0.6f}")
        logger.add_scalar('Loss/train', train_error, epoch)
        if val_dl is not None:
            validation_error = model.validate(val_dl)
            print(f"{epoch:05} Validation error: {validation_error: 0.6f}")
            logger.add_scalar('Loss/validation', train_error, epoch)
            
        if validation_error < best_validation_error or val_dl is None:
            best_validation_error = validation_error
            best_epoch = epoch
            model.save(save_folder / '{}_{:02d}'.format(nn_name, i) / '{:04d}.torch'.format(epoch), epoch)

    model.save(save_folder / '{}_{:02d}'.format(nn_name, i) / '{:04d}.torch'.format(epoch), epoch)
    return best_epoch

def compute_f1(pred, tg):
    '''Computes F1 score to evaluate accuracy of segmentation
    
    :param pred: Network prediction
    :type pred: :class:`np.ndarray`
    :param tg: Ground-truth segmentatin
    :type tg: :class:`np.ndarray`
    
    :return: F1 score
    :rtype: :class:`float`
    '''
    tp = np.count_nonzero(np.logical_and(pred == 1, tg == 1))
    fp = np.count_nonzero(np.logical_and(pred == 1, tg == 0))
    fn = np.count_nonzero(np.logical_and(pred == 0, tg == 1))
    
    if tp == 0:
        f1 = 0
    else:
        f1 = 2*float(tp) / (2*tp + fp + fn)
    
    return f1

def make_comparison(inp, tg, pred, fname):
    '''Visualizes the network prediction and compares it with ground-truth
    
    :param inp: Input image
    :type inp: :class:`np.ndarray`
    :param pred: Network prediction
    :type pred: :class:`np.ndarray`
    :param tg: Ground-truth segmentatin
    :type tg: :class:`np.ndarray`
    :param fname: File name to save the image
    :type fname: :class:`str`
    '''
    fig, ax = plt.subplots(1, 2, figsize=(18,9))
    
    segmented_im = np.zeros((*inp.shape, 3))
    display_im = (inp - inp.min()) / (inp.max() - inp.min()) * 1.
    
    for i in range(3):
        segmented_im[:,:,i] = display_im
    # Draw ground-truth boundary in red channel
    tg_fo = (tg == 2).astype(np.uint8)
    tg_map = morphology.binary_dilation(tg_fo) - tg_fo
    segmented_im[tg_map == 1, :] = 0.
    segmented_im[tg_map == 1, 0] = 1.
    # Draw prediction boundary in green channel
    pred_fo = (pred == 2).astype(np.uint8)
    pred_map = morphology.binary_dilation(pred_fo) - pred_fo
    segmented_im[pred_map == 1, :] = 0.
    segmented_im[pred_map == 1, 1] = 1.
    
    ax[0].imshow(display_im, cmap='gray')
    ax[0].set_title('Input image', fontsize=24)
    
    ax[1].imshow(segmented_im)
    ax[1].set_title('Segmentation comparison', fontsize=24)
    ax[1].text(40, 230, 'R = GT, G = Prediction', color='r', fontsize=16)
    f1 = compute_f1(pred_fo, tg_fo)
    ax[1].text(40, 250, "F1 score = {:.0%}".format(f1), color='r', fontsize=16)
    #plt.show()
    plt.savefig(fname)
    plt.clf()

def test(test_folder, nn_name, nn_num, save_comparison=False):
    '''Tests the network on a test dataset. F1 score is used as accuracy metric.
    
    :param test_folder: Test dataset folder
    :type test_folder: :class:`pathlib.PosixPath`
    :param nn_name: Network name
    :type nn_name: :class:`str`
    :param save_comparison: Flag to save images comparing network prediction and gt. Slows the testing process.
    :type save_comparison: :class:`bool`
    '''
    c_in = 2
    depth = 30
    width = 2
    dilations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    labels = [0, 1, 2]
    batch_size = 3
    
    save_folder = Path('../ckpt/')
    log_path = Path('../log/')
    res_folder = Path('./tmp_imgs/msd')
    if save_comparison:
        (res_folder / '{}_{:02d}'.format(nn_name, nn_num)).mkdir(exist_ok=True)
    
    model = mp.MSDSegmentationModel(c_in, len(labels), depth, width, dilations=dilations)
    fnames = sorted((save_folder / '{}_{:02d}'.format(nn_name, nn_num)).glob('*.torch'))
    print(fnames[-1])
    epoch = model.load(fnames[-1])
    test_input_glob = test_folder / 'val' / "inp" / "*.tiff"
    test_target_glob = test_folder / 'val' / "tg" / "*.tiff"
    ds = mp.ImageDataset(test_input_glob, test_target_glob)
    dl = DataLoader(ds, batch_size, shuffle=False)
    res = []
    
    for i, data in enumerate(dl):
        inp, tg = data
        output = model.net(inp.cuda())
        prediction = torch.max(output.data, 1).indices
        prediction_numpy = prediction.detach().cpu().numpy()
        print('inp', inp.detach().cpu().numpy().shape)
        print('tg', tg.detach().cpu().numpy().shape)
        inp_numpy = inp.detach().cpu().numpy()[:,0,:]
        tg_numpy = tg.detach().cpu().numpy().squeeze(1)
        for b in range(batch_size):
            output_index = i * batch_size + b
            if output_index < len(ds):
                acc = compute_f1(prediction_numpy[b], tg_numpy[b])
                res.append(acc)
                if save_comparison:
                    print(inp_numpy.shape)
                    print(tg_numpy.shape)
                    print(prediction_numpy.shape)
                    make_comparison(inp_numpy[b], tg_numpy[b], prediction_numpy[b], res_folder / '{}_{:02d}'.format(nn_name, nn_num) / 'output_{:03d}.png'.format(output_index))
                
    return np.array(res)

if __name__ == "__main__":
    train_folder = Path('/export/scratch2/vladysla/Data/Real/POD/datasets')
    test_folder = Path('/export/scratch2/vladysla/Data/Real/POD/datasets_var')
    data = [
        {
            'train' : [train_folder / '40kV_40W_100ms_10avg', train_folder / '90kV_45W_100ms_10avg'],
            'test' : [test_folder / '40kV_40W_100ms_10avg', test_folder / '90kV_45W_100ms_10avg']
        },
        {
            'train' : [train_folder / 'gen_40kV_40W_100ms_1avg', train_folder / 'gen_90kV_45W_100ms_1avg'],
            'test' : [test_folder / '40kV_40W_100ms_1avg', test_folder / '90kV_45W_100ms_1avg']
        },
        {
            'train' : [train_folder / 'gen_40kV_40W_50ms_1avg', train_folder / 'gen_90kV_45W_50ms_1avg'],
            'test' : [test_folder / '40kV_40W_50ms_1avg', test_folder / '90kV_45W_50ms_1avg']
        },
        {
            'train' : [train_folder / 'gen_40kV_40W_20ms_1avg', train_folder / 'gen_90kV_45W_20ms_1avg'],
            'test' : [test_folder / '40kV_40W_20ms_1avg', test_folder / '90kV_45W_20ms_1avg']
        },
                {
            'train' : [train_folder / 'gen_40kV_40W_100ms_1avg', train_folder / 'gen_90kV_45W_100ms_1avg'],
            'test' : [test_folder / 'gen_40kV_40W_100ms_1avg', test_folder / 'gen_90kV_45W_100ms_1avg']
        },
        {
            'train' : [train_folder / 'gen_40kV_40W_50ms_1avg', train_folder / 'gen_90kV_45W_50ms_1avg'],
            'test' : [test_folder / 'gen_40kV_40W_50ms_1avg', test_folder / 'gen_90kV_45W_50ms_1avg']
        },
        {
            'train' : [train_folder / 'gen_40kV_40W_20ms_1avg', train_folder / 'gen_90kV_45W_20ms_1avg'],
            'test' : [test_folder / 'gen_40kV_40W_20ms_1avg', test_folder / 'gen_90kV_45W_20ms_1avg']
        },
        {
            'train' : [train_folder / 'gen_40kV_40W_200ms_1avg', train_folder / 'gen_90kV_45W_200ms_1avg'],
            'test' : [test_folder / 'gen_40kV_40W_200ms_1avg', test_folder / 'gen_90kV_45W_200ms_1avg']
        },
        {
            'train' : [train_folder / 'gen_40kV_40W_500ms_1avg', train_folder / 'gen_90kV_45W_500ms_1avg'],
            'test' : [test_folder / 'gen_40kV_40W_500ms_1avg', test_folder / 'gen_90kV_45W_500ms_1avg']
        }
    ]
        
    train_folder = Path('/export/scratch2/vladysla/Data/Real/POD/datasets_var/msd_100ms/')
    test_folder = Path('/export/scratch2/vladysla/Data/Real/POD/datasets_var/msd_100ms/')
    nn_name = 'msd_100ms'
    test(test_folder, nn_name, 0, save_comparison=True)
    #train(train_folder, nn_name, 0)
    #for i in range(0, 10):
    #    train(train_folder, nn_name, i)
