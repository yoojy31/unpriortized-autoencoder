import argparse
import network
import dataset
import trainer

encoder_dict = {
    'basics00': network.encoder.BasicEncoder00,

    'limited00': network.encoder.LimitedEncoder00,
    'limited01': network.encoder.LimitedEncoder01,

    'latent_ar00': network.encoder.LatentAREncoder00,
    'latent_ar01': network.encoder.LatentAREncoder01,
    'latent_ar02': network.encoder.LatentAREncoder02,
    'latent_ar10': network.encoder.LatentAREncoder10,
    'latent_ar11': network.encoder.LatentAREncoder11,

    'latent_ar20': network.encoder.LatentAREncoder20,
    'latent_ar21': network.encoder.LatentAREncoder21,

    'latent_ar30': network.encoder.LatentAREncoder30,

    'latent_hier00': network.encoder.LatentHierarchicalEncoder00,

    'none': None,
    None: None,
}

decoder_dict = {
    'basics00': network.decoder.BasicDecoder00,
    'none': None,
    None: None,
}

sampler_dict = {
    # 5 layer, without batch norm, bias=False
    'ar00': network.sampler.ARSampler00,
    # 5 layer, with batch norm affine=True, bias=False
    'ar01': network.sampler.ARSampler01,
    # 5 layer, with batch norm affine=False, bias=False
    'ar02': network.sampler.ARSampler01,

    # 10 layer, without bath norm, bias=False
    'ar10': network.sampler.ARSampler10,

    # conv architecture, sliding window (with basic-trainer1)
    'ar20': network.sampler.ARSampler20,

    # hierarchical architecture, 3 layer each
    'hier00': network.sampler.HierarchicalSampler00,
    # hierarchical architecture, 6 layer each
    'hier10': network.sampler.HierarchicalSampler10,
    # hierarchical architecture, 6 layer each
    # batch norm affine=True train=True
    'hier11': network.sampler.HierarchicalSampler11,

    # parameter efficient hierarchical architecture
    'hier20': network.sampler.HierarchicalSampler20,

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
    'basics_ae0': trainer.ae.BasicAETrainer0,
    'mask_ae0': trainer.ae.MaskAETrainer0,

    # mlp sampler trainer
    'basics_sampler0': trainer.sampler.BasicSamplerTrainer0,
    # conv sampler trainer (sliding window)
    'basics_sampler1': trainer.sampler.BasicSamplerTrainer1,
    # conv sampler trainer (fixed position)
    'basics_sampler2': trainer.sampler.BasicSamplerTrainer2,

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
train_parser.add_argument('--lr', type=float, default=1e-3)
train_parser.add_argument('--lr_decay_rate', type=float, default=5e-1)
train_parser.add_argument('--lr_decay_intv', type=float, default=20)
train_parser.add_argument('--beta1', type=float, default=0.9)

train_parser.add_argument('--batch_size', type=int, default=128)
train_parser.add_argument('--img_size', type=int, default=64)
train_parser.add_argument('--img_ch', type=int, default=3)
train_parser.add_argument('--code_size', type=int, default=100)
train_parser.add_argument('--num_bin', type=int, default=100)

train_parser.add_argument('--print_intv', type=int, default=10)
train_parser.add_argument('--valid_intv', type=int, default=10)

train_parser.add_argument('--result_dir', type=str)
train_parser.add_argument('--snapshot_intv', type=int, default=10)
train_parser.add_argument('--snapshot_dir', type=str, default=None)
