import argparse
import dataset
import network

network_dict = {
    # basic autoencoder
    'ae00': network.ae.Autoencoder00,
    # basic autoencoder with tanh encoder
    'ae01': network.ae.Autoencoder01,
    # autoencoder with dropout
    'ae10': network.ae.Autoencoder10,
    # autoencoder with partial dropout
    'ae11': network.ae.Autoencoder11,
    # ladder autoencoder
    'lae00': network.ae.LadderAutoencoder00,

    'armdn00': network.armdn.ARMDN00,
    'gmmn': network.armdn.GMMN,

    'none': None,
    None: None,
}

dataset_dict = {
    'mnist': dataset.MNIST,
    'cifar': dataset.Cifar,
    'celeb': dataset.Celeb,
    'two': dataset.Two,

    'bsds': dataset.BSDS,
    'aug-bsds':dataset.AugmentedBSDS,
    'none': None,
    None: None,
}

#==================================================================================================
train_parser = argparse.ArgumentParser()
train_parser.add_argument('--devices', type=str, default='0')

train_parser.add_argument('--ae', type=str, default=None, help=network_dict.keys())
train_parser.add_argument('--armdn', type=str, default=None, help=network_dict.keys())

train_parser.add_argument('--z_size', type=int, default=64)
train_parser.add_argument('--static_z_size', type=int, default=64)
train_parser.add_argument('--z_dout_rate', type=float, default=0.5)
train_parser.add_argument('--z_mask_warm_up', type=int, default=0)

train_parser.add_argument('--n_gauss', type=int, default=20)
train_parser.add_argument('--tau', type=float, default=1.0)
train_parser.add_argument('--ordering', action='store_true')

train_parser.add_argument('--train_dataset', type=str, help=dataset_dict.keys())
train_parser.add_argument('--valid_dataset', type=str, help=dataset_dict.keys())
train_parser.add_argument('--train_set_path', type=str)
train_parser.add_argument('--valid_set_path', type=str)

train_parser.add_argument('--batch_size', type=int, default=128)
train_parser.add_argument('--img_size', type=int, default=64)
train_parser.add_argument('--img_ch', type=int, default=3)
train_parser.add_argument('--input_drop', type=float, default=0.0)
train_parser.add_argument('--patch_drop', action='store_true')

train_parser.add_argument('--init_epoch', type=int, default=0)
train_parser.add_argument('--max_epoch', type=int, default=30)
train_parser.add_argument('--lr', type=float, default=1e-3)
train_parser.add_argument('--lr_decay_rate', type=float, default=5e-1)
train_parser.add_argument('--lr_decay_epochs', type=str, default='')
train_parser.add_argument('--beta1', type=float, default=0.9)

train_parser.add_argument('--mse_w', type=float, default=1.0)
train_parser.add_argument('--perc_w', type=float, default=0.1)

train_parser.add_argument('--eval_epoch_intv', type=int, default=10)
train_parser.add_argument('--valid_iter_intv', type=int, default=50)

train_parser.add_argument('--result_dir', type=str)
train_parser.add_argument('--save_snapshot_epochs', type=str, default='')
train_parser.add_argument('--load_snapshot_dir', type=str, default=None)

#==================================================================================================
eval_parser = argparse.ArgumentParser()
eval_parser.add_argument('--devices', type=str, default='0')

eval_parser.add_argument('--ae', type=str, default=None, help=network_dict.keys())
eval_parser.add_argument('--armdn', type=str, default=None, help=network_dict.keys())
eval_parser.add_argument('--z_size', type=int, default=64)

eval_parser.add_argument('--n_gauss', type=int, default=20)
eval_parser.add_argument('--tau', type=float, default=1.0)

eval_parser.add_argument('--eval_dataset', type=str, help=dataset_dict.keys())
eval_parser.add_argument('--eval_set_path', type=str)

eval_parser.add_argument('--batch_size', type=int, default=128)
eval_parser.add_argument('--img_size', type=int, default=64)
eval_parser.add_argument('--img_ch', type=int, default=3)

eval_parser.add_argument('--load_snapshot_dir', type=str)
eval_parser.add_argument('--result_dir', type=str)

#==================================================================================================
preprocess_parser = argparse.ArgumentParser()
preprocess_parser.add_argument('--dataset', type=str)
preprocess_parser.add_argument('--train_set_path', type=str)
preprocess_parser.add_argument('--valid_set_path', type=str)
preprocess_parser.add_argument('--test_set_path', type=str)
preprocess_parser.add_argument('--result_dir', type=str)

#==================================================================================================
interp_parser = argparse.ArgumentParser()
interp_parser.add_argument('--ae', type=str, default=None, help=network_dict.keys())
interp_parser.add_argument('--z_size', type=int, default=64)
interp_parser.add_argument('--perc_w', type=float, default=0.0)

interp_parser.add_argument('--dataset', type=str)
interp_parser.add_argument('--dataset_path', type=str)
interp_parser.add_argument('--batch_size', type=int, default=128)
interp_parser.add_argument('--img_size', type=int, default=64)
interp_parser.add_argument('--img_ch', type=int, default=3)

interp_parser.add_argument('--pairs', type=str) # 1,2|3,5|0,10
interp_parser.add_argument('--load_snapshot_dir', type=str)
interp_parser.add_argument('--result_dir', type=str)

#==================================================================================================
plot_parser = argparse.ArgumentParser()
plot_parser.add_argument('--ae', type=str, default=None, help=network_dict.keys())
plot_parser.add_argument('--z_size', type=int, default=64)
plot_parser.add_argument('--perc_w', type=float, default=0.0)

plot_parser.add_argument('--dataset', type=str)
plot_parser.add_argument('--dataset_path', type=str)
plot_parser.add_argument('--batch_size', type=int, default=128)
plot_parser.add_argument('--max_iters', type=int, default=0)

plot_parser.add_argument('--img_size', type=int, default=64)
plot_parser.add_argument('--img_ch', type=int, default=3)

plot_parser.add_argument('--load_snapshot_dir', type=str)
plot_parser.add_argument('--result_dir', type=str)
