import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import random
from sklearn.metrics import confusion_matrix
import torch.utils.data as data


from model import *
from datasets import CIFAR10_truncated, CIFAR100_truncated, ImageFolder_custom

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass



def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_tinyimagenet_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_custom(datadir+'./train/', transform=transform)
    xray_test_ds = ImageFolder_custom(datadir+'./val/', transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)


def record_net_data_stats(y_train, net_dataidx_map, logdir):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(dataset, datadir, logdir, partition, n_parties, beta=0.4,
                   holdout_ratio=0.2, seed=0):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'tinyimagenet':
        X_train, y_train, X_test, y_test = load_tinyimagenet_data(datadir)

    n_train = y_train.shape[0]

    rng = np.random.RandomState(seed)

    # -------------------------
    # 1) 기존처럼 train partition 생성
    # -------------------------
    if partition == "homo" or partition == "iid":
        idxs = rng.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i].tolist() for i in range(n_parties)}

    elif partition == "noniid-labeldir" or partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset == 'cifar100':
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200

        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                rng.shuffle(idx_k)

                proportions = rng.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties)
                                        for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                cut = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

                splits = np.split(idx_k, cut)
                idx_batch = [idx_j + split.tolist() for idx_j, split in zip(idx_batch, splits)]

            min_size = min(len(idx_j) for idx_j in idx_batch)

        for j in range(n_parties):
            rng.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    # -------------------------
    # 2) client별 holdout split 생성 (train/local-test)
    # -------------------------
    net_dataidx_map_train = {}
    net_dataidx_map_test = {}
    for cid, idxs in net_dataidx_map.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)

        n_holdout = int(len(idxs) * holdout_ratio)
        # 너무 작아서 0 되면 최소 1은 떼도록(원하면 조정)
        if n_holdout == 0 and len(idxs) >= 2:
            n_holdout = 1

        test_idxs = idxs[:n_holdout].tolist()
        train_idxs = idxs[n_holdout:].tolist()

        net_dataidx_map_train[cid] = train_idxs
        net_dataidx_map_test[cid] = test_idxs

    # 로깅은 "학습에 쓰는 train split" 기준이 더 자연스러움
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map_train, logdir)

    return (X_train, y_train, X_test, y_test,
            net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts)


def get_trainable_parameters(net, device='cpu'):
    'return trainable parameter values as a vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    # print("net.parameter.data:", list(net.parameters()))
    paramlist = list(trainable)
    #print("paramlist:", paramlist)
    N = 0
    for params in paramlist:
        N += params.numel()
        # print("params.data:", params.data)
    X = torch.empty(N, dtype=torch.float64, device=device)
    X.fill_(0.0)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            X[offset:offset + numel].copy_(params.data.view_as(X[offset:offset + numel].data))
        offset += numel
    # print("get trainable x:", X)
    return X


def put_trainable_parameters(net, X):
    'replace trainable parameter values by the given vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    paramlist = list(trainable)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset + numel].data.view_as(params.data))
        offset += numel


def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu", multiloader=False):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device.type:
        criterion = nn.CrossEntropyLoss().cuda()
    loss_collector = []
    if multiloader:
        for loader in dataloader:
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(loader):
                    #print("x:",x)
                    #print("target:",target)
                    if device != 'cpu':
                        x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                    _, _, out = model(x)
                    if len(target)==1:
                        loss = criterion(out, target)
                    else:
                        loss = criterion(out, target)
                    _, pred_label = torch.max(out.data, 1)
                    loss_collector.append(loss.item())
                    total += x.data.size()[0]
                    correct += (pred_label == target.data).sum().item()

                    if device == "cpu":
                        pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                        true_labels_list = np.append(true_labels_list, target.data.numpy())
                    else:
                        pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                        true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
        avg_loss = sum(loss_collector) / len(loss_collector)
    else:
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(dataloader):
                #print("x:",x)
                if device != 'cpu':
                    x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                _,_,out = model(x)
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                loss_collector.append(loss.item())
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
            avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss

    return correct / float(total), avg_loss

def compute_loss(model, dataloader, device="cpu"):
    was_training = False
    if model.training:
        model.eval()
        was_training = True
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device.type:
        criterion = nn.CrossEntropyLoss().cuda()
    loss_collector = []
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            if device != 'cpu':
                x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            _,_,out = model(x)
            loss = criterion(out, target)
            loss_collector.append(loss.item())

        avg_loss = sum(loss_collector) / len(loss_collector)

    if was_training:
        model.train()

    return avg_loss



def save_model(model, model_index, args):
    logger.info("saving local model-{}".format(model_index))
    with open(args.modeldir + "trained_local_model" + str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return


def load_model(model, model_index, device="cpu"):
    #
    with open("trained_local_model" + str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    if device == "cpu":
        model.to(device)
    else:
        model.cuda()
    return model


def get_dataloader(
    dataset,
    datadir,
    train_bs,
    test_bs,
    dataidxs=None,          # client train indices
    test_dataidxs=None,     # client test indices (optional)
    noise_level=0
):
    
    if dataset in ("cifar10", "cifar100"):
        if dataset == "cifar10":
            dl_obj = CIFAR10_truncated
            normalize = transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            )
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode="reflect"
                ).data.squeeze()),
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=noise_level),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        else:  # cifar100
            dl_obj = CIFAR100_truncated
            normalize = transforms.Normalize(
                mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404],
            )
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize,
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        # ---- train subset (client train) ----
        train_ds = dl_obj(
            datadir,
            dataidxs=dataidxs,
            train=True,
            transform=transform_train,
            download=True,
        )

        # ---- test set ----
        # 1) if client-specific test indices are provided: build subset from training split (train=True)
        # 2) else: use standard global test split (train=False)
        if test_dataidxs is not None:
            test_ds = dl_obj(
                datadir,
                dataidxs=test_dataidxs,
                train=True,                # IMPORTANT: because indices refer to training split
                transform=transform_test,
                download=True,
            )
        else:
            test_ds = dl_obj(
                datadir,
                train=False,
                transform=transform_test,
                download=True,
            )

        train_dl = data.DataLoader(
            dataset=train_ds,
            batch_size=train_bs,
            drop_last=True,
            shuffle=True,
        )
        test_dl = data.DataLoader(
            dataset=test_ds,
            batch_size=test_bs,
            shuffle=False,
        )

        return train_dl, test_dl, train_ds, test_ds

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")