import datetime
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data as Data
from tqdm import trange, tqdm
from model import GCN, CreateFeature
from utils import get_data_split, get_acc, use_cuda, load_data_set, setup_seed

device = use_cuda()


def train(args):
    # setup_seed(args.seed)

    path = '_'.join(['tmp/' + args.dataset,
                     'lr', str(args.lr_cl), str(args.lr_gcn),
                     'epoch', str(args.n_epochs_gcn), str(args.n_epochs_linear),
                     'wd', str(args.wd_cl), str(args.wd_gcn),
                     'cl', str(args.n_dim_cl), str(args.n_bsize_cl)
                     ])
    [c_train, c_val] = args.train_val_class
    idx, labellist, G, bow_feature, csd_matrix, lda_feature, node_feature, kg_feature = load_data_set(args.dataset)
    idx_train, idx_test, idx_val = get_data_split(c_train=c_train, c_val=c_val, idx=idx, labellist=labellist)

    y_true = np.array([int(temp[0]) for temp in labellist])  # [n, 1]
    y_true = torch.from_numpy(y_true).type(torch.LongTensor).to(device)
    bow_feature = bow_feature.to(device)
    lda_feature = lda_feature.to(device)
    node_feature = node_feature.to(device)
    kg_feature = kg_feature.to(device)
    G = G.to(device)
    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    cl_pth_path = path + '_cl.pth'
    cl = CreateFeature(bow_feature.shape[-1], node_feature.shape[-1], kg_feature.shape[-1], args.n_hidden,
                       args.n_dim_cl)

    cl = cl.to(device)
    if not os.path.exists(cl_pth_path):
        cl_dataset = Data.TensorDataset(bow_feature, node_feature, kg_feature, lda_feature)
        cl_loader = Data.DataLoader(dataset=cl_dataset, batch_size=args.n_bsize_cl)
        optimiser_cl = torch.optim.Adam(cl.parameters(), lr=args.lr_cl, weight_decay=args.wd_cl)
        scheduler_cl = torch.optim.lr_scheduler.StepLR(optimiser_cl, step_size=10, gamma=0.5)
        cl.train()
        with tqdm(total=args.n_epochs_cl) as pbar:
            for epoch in range(args.n_epochs_cl + 1):
                loss_num = 0
                for batch, (bow, node, kg, lda) in enumerate(cl_loader):
                    optimiser_cl.zero_grad()
                    _, loss = cl(bow, node, kg, lda)
                    if epoch % 10 == 0:
                        loss_num += loss.item()
                    loss.backward()
                    optimiser_cl.step()
                    pbar.set_postfix_str(str(batch) + '/' + str(len(cl_loader)))
                if epoch % 10 == 0:
                    print('\n', 'CL loss: ', loss_num / len(cl_loader))
                pbar.update(1)
                scheduler_cl.step()
        torch.save(cl.state_dict(), cl_pth_path)
    else:
        cl.load_state_dict(torch.load(cl_pth_path, map_location=device))

    cl.eval()
    input_feature, _ = cl(bow_feature, node_feature, kg_feature, lda_feature)
    input_feature = input_feature.detach().to(device)

    model = GCN(input_feature.shape[1], args.n_hidden, args.dropout).to(device)

    csd_matrix = csd_matrix.to(device)

    criterion = nn.CrossEntropyLoss()
    optimiser_gcn = torch.optim.Adam(model.graphconv.parameters(), lr=args.lr_gcn, weight_decay=args.wd_gcn)
    optimiser_linear = torch.optim.Adam(model.linear.parameters(), lr=args.lr_linear, weight_decay=args.wd_linear)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser_linear, step_size=1000, gamma=0.8)
    for epoch in trange(args.n_epochs_linear):
        model.train()
        optimiser_gcn.zero_grad()
        optimiser_linear.zero_grad()

        preds = model(input_feature, G, csd_matrix)
        loss = criterion(preds[idx_train], y_true[idx_train])

        if epoch % 100 == 0:
            train_acc = get_acc(preds[idx_train], y_true[idx_train], c_train=c_train, c_val=c_val, model='train')
            test_acc = get_acc(preds[idx_test], y_true[idx_test], c_train=c_train, c_val=c_val, model='test')
            print('Loss:', loss.item(), 'Train_acc:', train_acc, 'Test_acc:', test_acc)
            model.eval()
            preds = model(input_feature, G, csd_matrix)
            test_acc = get_acc(preds[idx_test], y_true[idx_test], c_train=c_train, c_val=c_val, model='test')
            print('Evaluation!', 'Test_acc:', test_acc, "+++")

        loss.backward()
        if epoch < args.n_epochs_gcn:
            optimiser_gcn.step()
        optimiser_linear.step()
        scheduler.step()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MODEL')
    parser.add_argument("--dataset", type=str, default='Cora_enrich', help="dataset")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--train-val-class", type=int, nargs='*', default=[3, 0],
                        help="the first #train_class and #validation classes")
    parser.add_argument("--lr-cl", type=float, default=0.001, help="learning rate for cl")
    parser.add_argument("--lr-gcn", type=float, default=0.0001, help="learning rate for gcn")
    parser.add_argument("--lr-linear", type=float, default=0.01, help="learning rate for linear")
    parser.add_argument("--n-epochs-cl", type=int, default=50, help="number of training epochs for cl")
    parser.add_argument("--n-epochs-gcn", type=int, default=10, help="number of training epochs for gcn")
    parser.add_argument("--n-epochs-linear", type=int, default=10000, help="number of training epochs for linear")
    parser.add_argument("--n-hidden", type=int, default=128, help="number of hidden layers")
    parser.add_argument("--wd-cl", type=float, default=0, help="Weight for L2 loss in cl")
    parser.add_argument("--wd-gcn", type=float, default=5e-4, help="Weight for L2 loss in gcn")
    parser.add_argument("--wd-linear", type=float, default=5e-4, help="Weight for L2 loss in linear")
    parser.add_argument("--n-dim-cl", type=int, default=256, help="output dimension of cl")
    parser.add_argument("--n-bsize-cl", type=int, default=128, help="Batch size for cl")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    train(args)
