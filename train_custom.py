import warnings
import numpy as np
import os
import utils
import dataloader
import pandas as pd
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
import wandb
from tqdm import tqdm
import torch
from torch import nn
import segmentation_models_pytorch as smp
from torchsummary import summary
import math
import time
from model import UNet

warnings.filterwarnings('ignore')
wandb.init(project='Gland_Seg', entity='glaseg', config='config/config.yaml')
config = wandb.config
tabular_data = pd.read_csv(config.csv)
ds_dict = dataloader.get_split_fold(tabular_data)
patch_size = eval(config.patch_size)
batch_size = config.batch_size
epochs = config.epochs
tr_transforms = dataloader.get_train_transform(patch_size, prob=config.aug_prob)
train_dl = dataloader.DataLoader(data=ds_dict['train_ds'], batch_size=batch_size, patch_size=patch_size,
                                 num_threads_in_multithreaded=4, seed_for_shuffle=5243,
                                 return_incomplete=False, shuffle=True, infinite=True)
train_gen = MultiThreadedAugmenter(train_dl, tr_transforms, num_processes=4,
                                   num_cached_per_queue=2,
                                   seeds=None, pin_memory=False)
val_dl = dataloader.DataLoader(data=ds_dict['testA_ds'], batch_size=batch_size, patch_size=patch_size,
                               num_threads_in_multithreaded=1, seed_for_shuffle=5243,
                               return_incomplete=False, shuffle=True, infinite=True)

# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
# define model
model = UNet(in_channels=3, init_filter=64, depth=3).to(device)
summary(model, (3, 512, 512))
optimizer = eval(config.optimizer)(model.parameters(), lr=float(config.learning_rate))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20,
                                                       factor=0.1)
# xent = nn.BCELoss()
dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
# label smoothing
xent = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.1)


def custom_loss(pred, target):
    xent_l = xent(pred, target)
    dice_l = dice_loss(pred, target)
    loss = xent_l + dice_l
    return loss, xent_l, dice_l


def train(model, optimizer):
    # total number of training batches
    num_batches = math.ceil(len(ds_dict['train_ds']['img_npy'])/batch_size)
    model.train()
    batch_xent_l = []
    batch_dice_l = []
    batch_loss = []
    print("Training...")
    for i in tqdm(range(num_batches)):
        train_batch = next(train_gen)
        imgs = train_batch['data']
        segs = train_batch['seg']
        # normalization
        imgs = utils.min_max_norm(imgs)
        # binarisation
        segs = np.where(segs > 0., 1.0, 0.).astype('float32')
        segs = np.expand_dims(segs[:, 0, :, :], 1)
        imgs, segs = torch.from_numpy(imgs).to(device), torch.from_numpy(segs).to(device)
        # Compute loss
        pred = model(imgs)
        loss, xent_l, dice_l = custom_loss(pred, segs)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # batch losses
        batch_xent_l.append(xent_l)
        batch_dice_l.append(dice_l)
        batch_loss.append(loss)
    # apply sigmoid to masking
    segs = nn.Sigmoid()(segs)
    # taking the average along the batch
    loss = torch.mean(torch.as_tensor(batch_loss)).item()
    avg_xent_l = torch.mean(torch.as_tensor(batch_xent_l)).item()
    avg_dice_l = torch.mean(torch.as_tensor(batch_dice_l)).item()

    return {'loss': loss, 'xent_l': avg_xent_l, 'dice_l': avg_dice_l,
            'imgs': imgs.cpu().detach().numpy(),
            'segs': segs.cpu().detach().numpy(),
            'pred': pred.cpu().detach().numpy()}


def test(model):
    num_batches = math.ceil(len(ds_dict['testA_ds']['img_npy']) / batch_size)
    model.eval()
    # no need back prop for testing set
    batch_xent_l = []
    batch_dice_l = []
    batch_loss = []
    print("Testing...")
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            val_batch = next(val_dl)
            imgs = val_batch['data']
            segs = val_batch['seg']
            # normalization
            imgs = utils.min_max_norm(imgs)
            # binarisation
            segs = np.where(segs > 0., 1.0, 0.).astype('float32')
            segs = np.expand_dims(segs[:, 0, :, :], 1)
            imgs, segs = torch.from_numpy(imgs).to(device), torch.from_numpy(segs).to(device)
            # Compute loss
            pred = model(imgs)
            loss, xent_l, dice_l = custom_loss(pred, segs)
            # batch losses
            batch_xent_l.append(xent_l)
            batch_dice_l.append(dice_l)
            batch_loss.append(loss)
        # apply sigmoid to masking
        segs = nn.Sigmoid()(segs)
        # taking the average along the batch
        loss = torch.mean(torch.as_tensor(batch_loss)).item()
        avg_xent_l = torch.mean(torch.as_tensor(batch_xent_l)).item()
        avg_dice_l = torch.mean(torch.as_tensor(batch_dice_l)).item()
    return {'loss': loss, 'xent_l': avg_xent_l, 'dice_l': avg_dice_l,
            'imgs': imgs.cpu().detach().numpy(),
            'segs': segs.cpu().detach().numpy(),
            'pred': pred.cpu().detach().numpy()}


start = time.time()

def main():
    current_total_loss = 1000
    current_dice_loss = 1000
    for e in range(1, epochs+2):
        print("Epcohs:", e)
        train_output = train(model, optimizer)
        test_output = test(model)
        scheduler.step(test_output['loss'])
        print("Training Outputs: ")
        print("Total loss: {:.2f}, BCE: {:.2f}, Dice Loss: {:.2f}".format(train_output['loss'], train_output['xent_l'], train_output['dice_l']))
        print("-"*100)
        print("Validation Outputs: ")
        print("Total loss: {:.2f}, BCE: {:.2f}, Dice Loss: {:.2f}".format(test_output['loss'], test_output['xent_l'], test_output['dice_l']))
        # logging
        wandb.log({"Train_total_loss": train_output['loss'], "Val_total_loss": test_output['loss']}, step=e)
        wandb.log({"Train_BCE_loss": train_output['xent_l'], "Val_BCE_loss": test_output['xent_l']}, step=e)
        wandb.log({"Train_dice_loss": train_output['dice_l'], "Val_dice_loss": test_output['dice_l']}, step=e)
        wandb.log({"Learning rate": optimizer.param_groups[0]["lr"]}, step=e)
        if e%10==0:
            # threshold sigmoid output with 0.5
            pred_thr = np.where(test_output['pred']>0.5, 1.0, 0.0)
            # sample a dataset from the batch for visualization purpose
            imgs = [test_output['imgs'][0, 0, :, :], test_output['segs'][0, 0, :, :], pred_thr[0, 0, :, :]]
            captions = ['Gland Image', 'Masking', 'Prediction']
            fig = utils.plot_comparison(imgs, captions, plot=False, n_col=len(imgs),
                                  figsize=(12, 12), cmap='gray')
            wandb.log({"Validation Dataset Output Sample": wandb.Image(fig)}, step=e)

        # save model
        weights_dir = '/home/kevinteng/Desktop/weights/'
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        base_path = os.path.split(weights_dir)[0]
        if test_output['loss'] < current_total_loss:
            current_total_loss = test_output['loss']
            torch.save(model.state_dict(), weights_dir+'best_loss_{}.pth'.format(e))
            wandb.save(os.path.join(weights_dir, 'best_loss_{}.pth'.format(e)), base_path=base_path)
        if test_output['dice_l'] < current_dice_loss:
            current_dice_loss = 1-test_output['dice_l']
            torch.save(model.state_dict(), weights_dir+'best_dice_{}.pth'.format(e))
            wandb.save(os.path.join(weights_dir, 'best_dice_{}.pth'.format(e)), base_path=base_path)
        print()

    print("Model training runtime: {} mins".format((time.time() - start)/60.0))


if __name__ == '__main__':
    main()


