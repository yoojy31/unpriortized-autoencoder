import os
import time
import itertools
from tqdm import tqdm
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import options
import utils
import eval_ae

def train():
    # Create result directories--------------------------------------------------------
    result_dir_dict = utils.create_result_dir(args.result_dir)

    # Create network and optimizer-----------------------------------------------------
    autoencoder_cls = options.network_dict[args.autoencoder]
    autoencoder = None if autoencoder_cls is None else autoencoder_cls(args)
    autoencoder.build()
    params = autoencoder.parameters()
    optim = torch.optim.Adam(params, lr=args.lr, betas=(args.beta1, 0.999))

    # Load models and optimizer and use cuda-------------------------------------------
    if args.load_snapshot_dir is not None:
        autoencoder.load(args.load_snapshot_dir)
        load_optim(optim, args.load_snapshot_dir)
    autoencoder.cuda()

    # Load dataset---------------------------------------------------------------------
    dataset_cls = options.dataset_dict[args.dataset]
    train_dataset = dataset_cls(args, args.train_set_path)
    valid_dataset = dataset_cls(args, args.valid_set_path)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4)
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=1)

    # Create logger--------------------------------------------------------------------
    train_logger = SummaryWriter(result_dir_dict['train'])
    valid_logger = SummaryWriter(result_dir_dict['valid'])
    eval_logger = SummaryWriter(result_dir_dict['eval'])

    # Training-------------------------------------------------------------------------
    cur_lr = args.lr
    num_batch = train_data_loader.__len__()
    save_input_imgs = next(iter(valid_data_loader))['image'].cuda()
    for e in range(args.init_epoch, args.max_epoch + 1):

        # Save image result------------------------------------------------------------
        autoencoder.train(mode=False)
        save_output_imgs = autoencoder.forward(save_input_imgs, 'all')
        utils.save_img_batch(
            result_dir_dict['img'], save_input_imgs,
            valid_dataset.inv_preprocessing, 'input_img')
        utils.save_img_batch(
            result_dir_dict['img'], save_output_imgs,
            valid_dataset.inv_preprocessing, 'output_img')

        # Evalutaion-------------------------------------------------------------------
        if e % args.eval_epoch_intv == 0:
            eval_mse_loss = eval_ae.evalate(autoencoder, valid_data_loader)
            global_step = e * num_batch
            eval_logger.add_scalar('mse_loss', eval_mse_loss, global_step)

        # Learning rate decay----------------------------------------------------------
        if e in args.lr_decay_epochs:
            cur_lr *= args.lr_decay_rate
            for param_group in optim.param_groups:
                param_group['lr'] = cur_lr
            pre_lr = cur_lr / args.lr_decay_rate
            print('learning rate decay: %f --> %f' % (pre_lr, cur_lr))

        # Save snapshot----------------------------------------------------------------
        if e in args.save_snapshot_epochs:
            snapshot_dir = os.path.join(
                result_dir_dict['snapshot'], 'epoch %d' % e)
            utils.make_dir(snapshot_dir)
            autoencoder.save(snapshot_dir)
            save_optim(optim, snapshot_dir)

        # Break loop-------------------------------------------------------------------
        if e == args.max_epoch:
            break

        # Train one epoch--------------------------------------------------------------
        valid_set_cycle = itertools.cycle(valid_data_loader)
        train_loader_pbar = tqdm(train_data_loader)
        accum_batching_time = 0
        accum_training_time = 0

        # Get batch dict---------------------------------------------------------------
        t1_1 = time.time()
        for b, train_batch_dict in enumerate(train_loader_pbar):
            t1_2 = time.time()

            # Train one step-----------------------------------------------------------
            t2_1 = time.time()
            x = train_batch_dict['image'].cuda()
            x.requires_grad_(True)

            autoencoder.train(mode=True)
            x_ = autoencoder.forward(x, 'all')
            train_mse_loss = F.mse_loss(x_, x)

            optim.zero_grad()
            train_mse_loss.backward()
            optim.step()
            t2_2 = time.time()

            # Validation---------------------------------------------------------------
            if b % args.valid_iter_intv == 0:
                valid_batch_dict = next(valid_set_cycle)
                x = valid_batch_dict['image'].cuda()
                x.requires_grad_(False)

                autoencoder.train(mode=False)
                x_ = autoencoder.forward(x, 'all')
                valid_mse_loss = F.mse_loss(x_, x)

            # Set description and write log--------------------------------------------
            accum_batching_time += (t1_2 - t1_1)
            accum_training_time += (t2_2 - t2_1)
            train_loader_pbar.set_description(
                '[training] epoch: %d/%d, ' % (e, args.max_epoch) +
                'batching: %.3fs/b, training: %.3fs/b |' % \
                (accum_batching_time / (b + 1), accum_training_time / (b + 1)))

            global_step = e * num_batch + b
            train_logger.add_scalar('mse_loss', train_mse_loss.item(), global_step)
            valid_logger.add_scalar('mse_loss', valid_mse_loss.item(), global_step)
            t1_1 = time.time()

    # Close logger---------------------------------------------------------------------
    train_logger.close()
    valid_logger.close()
    eval_logger.close()


# Load optim---------------------------------------------------------------------------
def load_optim(optim, load_dir):
    file_name = optim.__class__.__name__
    optim_path = os.path.join(load_dir, file_name)
    if os.path.exists(optim_path):
        optim.load_state_dict(torch.load(optim_path))
    else:
        raise FileNotFoundError(load_dir)


# Save optim---------------------------------------------------------------------------
def save_optim(optim, save_dir):
    if os.path.exists(save_dir):
        if not os.path.isdir(save_dir):
            raise IsADirectoryError(save_dir)
        else:
            pass
        file_name = optim.__class__.__name__
        optim_path = os.path.join(save_dir, file_name)
        torch.save(optim.state_dict(), optim_path)
    else:
        raise FileNotFoundError(save_dir)


if __name__=='__main__':
    args = options.train_ae_parser.parse_args()
    args.lr_decay_epochs = [int(e) for e in args.lr_decay_epochs.split(',')]
    args.save_snapshot_epochs = [int(e) for e in args.save_snapshot_epochs.split(',')]
    train()
