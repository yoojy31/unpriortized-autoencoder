import argparse
import dataset
import network

network_dict = {
    'autoencoder00': network.autoencoder.Autoencoder00,
    'none': None,
    None: None,
}

dataset_dict = {
    'mnist': dataset.MNIST,
    'celeb': dataset.Celeb,
    'none': None,
    None: None,
}

#==================================================================================================
train_ae_parser = argparse.ArgumentParser()
train_ae_parser.add_argument('--cur_device', type=int, default=0)

train_ae_parser.add_argument('--autoencoder', type=str, default=None, help=network_dict.keys())
train_ae_parser.add_argument('--z_size', type=int, default=100)

train_ae_parser.add_argument('--dataset', type=str, help=dataset_dict.keys())
train_ae_parser.add_argument('--train_set_path', type=str)
train_ae_parser.add_argument('--valid_set_path', type=str)

train_ae_parser.add_argument('--batch_size', type=int, default=128)
train_ae_parser.add_argument('--img_size', type=int, default=64)
train_ae_parser.add_argument('--img_ch', type=int, default=3)

train_ae_parser.add_argument('--init_epoch', type=int, default=0)
train_ae_parser.add_argument('--max_epoch', type=int, default=30)
train_ae_parser.add_argument('--lr', type=float, default=1e-3)
train_ae_parser.add_argument('--lr_decay_rate', type=float, default=5e-1)
train_ae_parser.add_argument('--lr_decay_epochs', type=str, default='')
train_ae_parser.add_argument('--beta1', type=float, default=0.9)

train_ae_parser.add_argument('--eval_epoch_intv', type=int, default=10)
train_ae_parser.add_argument('--valid_iter_intv', type=int, default=50)

train_ae_parser.add_argument('--result_dir', type=str)
train_ae_parser.add_argument('--save_snapshot_epochs', type=str, default='')
train_ae_parser.add_argument('--load_snapshot_dir', type=str, default=None)
