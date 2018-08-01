import argparse
import network
import dataset
import trainer

encoder_dict = {
    'basics00': network.encoder.BasicsEncoder00,
    'ar_mlp00': network.encoder.ARMLPEncoder00,
    'ar_mlp01': network.encoder.ARMLPEncoder01,
    'none': None,
    None: None,
}

decoder_dict = {
    'basics00': network.decoder.BasicsDecoder00,
    'none': None,
    None: None,
}

sampler_dict = {
    'none': None,
    None: None,
}

discrim_dict = {
    'none': None,
    None: None,
}

dataset_dict = {
    'mnist': dataset.MNIST,
    'celeb': dataset.Celeb,
    'none': None,
    None: None,
}

trainer_dict = {
    'basics_ae0': trainer.ae.BagicsAETrainer0,
    'none': None,
    None: None,
}

#==================================================================================================
train_parser = argparse.ArgumentParser()
train_parser.add_argument('--cur_device', type=int, default=0)

train_parser.add_argument('--encoder', type=str, default=None, help=encoder_dict.keys())
train_parser.add_argument('--decoder', type=str, default=None, help=decoder_dict.keys())
train_parser.add_argument('--sampler', type=str, default=None, help=sampler_dict.keys())
train_parser.add_argument('--discrim', type=str, default=None, help=discrim_dict.keys())

train_parser.add_argument('--dataset', type=str, help=dataset_dict.keys())
train_parser.add_argument('--train_set_path', type=str)
train_parser.add_argument('--valid_set_path', type=str)

train_parser.add_argument('--trainer', type=str, help=trainer_dict.keys())
train_parser.add_argument('--start_epoch', type=int, default=0)
train_parser.add_argument('--finish_epoch', type=int, default=30)
train_parser.add_argument('--lr', type=float, default=2e-4)
train_parser.add_argument('--beta1', type=float, default=0.9)

train_parser.add_argument('--batch_size', type=int, default=128)
train_parser.add_argument('--img_h', type=int, default=32)
train_parser.add_argument('--img_w', type=int, default=32)
train_parser.add_argument('--img_ch', type=int, default=3)
train_parser.add_argument('--code_size', type=int, default=100)
train_parser.add_argument('--num_bin', type=int, default=100)

train_parser.add_argument('--print_intv', type=int, default=10)
train_parser.add_argument('--valid_intv', type=int, default=10)

train_parser.add_argument('--result_dir', type=str)
train_parser.add_argument('--snapshot_intv', type=int, default=10)
train_parser.add_argument('--snapshot_dir', type=str, default=None)
