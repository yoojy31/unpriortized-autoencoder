import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import utils
import option

two_labels = ('0', '1')
mnist_labels = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
tableau20 = np.array(
    ((31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
    (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
    (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
    (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
    (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229))) / 255

def main():
    utils.make_dir(args.result_dir)

    ae = option.network_dict[args.ae](args)
    ae.build()
    if args.load_snapshot_dir is not None:
        assert ae.load(args.load_snapshot_dir)
    ae.cuda()
    ae.train(mode=False)

    dataset_cls = option.dataset_dict[args.dataset]
    dataset = dataset_cls(args, args.dataset_path, True)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False)

    z_list = list()
    y_list = list()
    for i, batch_dict in enumerate(data_loader):
        if (args.max_iters > 0) and (i >= args.max_iters):
            break

        x = batch_dict['image'].cuda()
        x.requires_grad_(False)

        z = ae.forward(x, forward_type='encoder')
        z = z.squeeze()
        z = z.cpu().detach().numpy()
        z_list.append(z)

        if 'label' in batch_dict.keys():
            y = batch_dict['label'].detach().numpy()
            y_list.append(y)

    z = np.concatenate(z_list, axis=0)
    if len(y_list) > 0:
        y = np.concatenate(y_list, axis=0)#.astype(np.uint8)
    else:
        y = None

    if args.dataset == 'mnist' or args.dataset == 'cifar':
        labels = mnist_labels
    elif args.dataset == 'two':
        labels = two_labels
    else:
        labels = None

    print('start pca')
    draw_scatter_plot(z, y, save_dir=args.result_dir, projection='PCA', colors=tableau20, labels=labels)
    print('save pca result\n')

    print('start tsne')
    subset = int(z.shape[0] / 1)
    draw_scatter_plot(z[:subset], y[:subset], save_dir=args.result_dir, projection='TSNE', colors=tableau20, labels=labels)
    print('save tsne result')

def draw_scatter_plot(x, y, save_dir, projection='PCA', colors={'gray'}, labels=None):
    if x.shape[1] == 2:
        projection = 'none'
    elif projection == 'PCA':
        pca = PCA(n_components=2)
        x = pca.fit_transform(x)
    elif projection == 'TSNE':
        tsne = TSNE(n_components=2, perplexity=100, learning_rate=200, n_iter=5000)
        x = tsne.fit_transform(x)
    else:
        assert projection in ('PCA', 'TSNE')

    for i in range(x.shape[0]):
        # color = (colors[int(y[i])][0] / 255, colors[int(y[i])][1] / 255, colors[int(y[i])][2] / 255)
        plt.plot(x[i, 0], x[i][1], color=colors[int(y[i])], marker='o', markersize=2)

    patch = []
    if labels is not None:
        for i in range(len(labels)):
            label = labels[i]
            patch.append(mpatches.Patch(color=colors[i], label=label))
        plt.legend(handles=patch)

    plt.savefig(os.path.join(save_dir, 'scatter-plot_%s.png') % projection)
    # plt.show()

if __name__ == "__main__":
    args = option.plot_parser.parse_args()
    main()
