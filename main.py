import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
from datetime import datetime
import random
from collections import defaultdict, deque
import copy
import torch.nn.functional as F
from eval_personalization import evaluate_personalization, evaluate_generalization_head_avg

from model import *
from utils import *

# 저장 폴더 설정 (드라이브에 fed_runs폴더 하위에 기록)
def get_run_dir(args):
    base = "/content/drive/Shareddrives/sail_seminar/3.experiments/fed_runs" if os.path.exists("/content/drive/Shareddrives/sail_seminar/3.experiments") else "./fed_runs"
    os.makedirs(base, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.alg}_{args.dataset}_P{args.n_parties}_R{args.comm_round}_{stamp}"
    run_dir = os.path.join(base, exp_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# 실험 저장
def save_experiment(run_dir, args, global_model, g_acc, p_acc):
    # 모델 저장
    torch.save(global_model.state_dict(),
               os.path.join(run_dir, "global_model_final.pth"))

    # args 저장
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2) # args를 dict로(vars) -> json으로(json.dump) 

    # 결과 저장
    results = {
        "generalization_acc": float(g_acc),
        "personalization_acc": float(p_acc)
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2) 

def save_ckpt(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)

def load_ckpt(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)

# 완전한 재현을 위한 난수 상태 저장 및 복원
def capture_rng_state(): # 현재 모든 랜덤 상태 저장
    state = {
        "py_random": random.getstate(),
        "np_random": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    return state

def restore_rng_state(state): # 저장했던 랜덤 상태 복구
    random.setstate(state["py_random"])
    np.random.set_state(state["np_random"])
    torch.random.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and "cuda_rng" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng"])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', ')))) # 문자열을 정수 인덱스로
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2, help='number of workers in a distributed cluster') # 전체 클라이언트 수
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication round')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength") # 가중치 감쇠 (옵티마이저 레벨)
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon or fedcap')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=100, help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    # parser.add_argument('--load_model_round', type=int, default=None, help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model') # 0으로 세팅해서 ModelFedCon/ModelFedCon_noheader 사용
    parser.add_argument('--loss', type=str, default='contrastive') # 이것도
    parser.add_argument('--save_model',type=int,default=0)
    parser.add_argument('--use_project_head', type=int, default=1) # 프로젝션 헤드 안 쓸 때 0으로 세팅
    # parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    # ----- FedBABU / FedCAP evaluation hyperparameters -----
    parser.add_argument('--Kg', type=int, default=50,
                        help='number of head fine-tuning steps for generalization evaluation')
    parser.add_argument('--Kp', type=int, default=50,
                        help='number of head fine-tuning steps for personalization evaluation')
    parser.add_argument('--head_lr', type=float, default=0.01,
                        help='learning rate for head fine-tuning')
    parser.add_argument('--eval_every', type=int, default=0,
                        help='evaluate personalization every N rounds (0 = only final round)')
    parser.add_argument('--resume_ckpt', type=str, default=None, help='path to checkpoint .pth')
    parser.add_argument('--ckpt_every', type=int, default=5, help='save checkpoint every N rounds')

    args = parser.parse_args()
    
    run_dir = get_run_dir(args)
    print("Saving to:", run_dir)
    
    return args, run_dir


def init_nets(net_configs, n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 26
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset == 'xray':
        n_classes = 2
    if args.normal_model:
        for net_i in range(n_parties):
            if args.model == 'simple-cnn':
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.to(device)
            nets[net_i] = net
    else: # normal_model: 0
        for net_i in range(n_parties):
            if args.use_project_head:
                net = ModelFedCon(args.model, args.out_dim, n_classes, net_configs)
            else:
                net = ModelFedCon_noheader(args.model, args.out_dim, n_classes, net_configs)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.to(device)
            nets[net_i] = net # 클라이언트별 모델 초기화

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type

def to_device(model, device):
    return model.to(device)

# DataParallel로 감싸진 모델이면 안에 있는 진짜 모델을 꺼내주는 함수
def unwrap_dp(model):
    return model.module if hasattr(model, "module") else model

def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    # net = nn.DataParallel(net)
    # net.cuda()
    net = to_device(net, device)
    logger.info('Training network %s' % str(net_id)) # 클라이언트 인덱스
    logger.info('n_training: %d' % len(train_dataloader)) 
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc,_ = compute_accuracy(net, train_dataloader, device=device)

    test_acc, conf_matrix,_ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _,_,out = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
            
        if epoch % 10 == 0:
            train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

            logger.info('>> Training accuracy: %f' % train_acc)
            logger.info('>> Test accuracy: %f' % test_acc)

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')

    logger.info(' ** Training complete **')
    return train_acc, test_acc


def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, args,
                      device="cpu"):
    global_net.to(device)
    # net = nn.DataParallel(net)
    # net.cuda()
    # else:
    net.to(device)
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    global_weight_collector = list(global_net.to(device).parameters())


    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _,_,out = net(x)
            loss = criterion(out, target)

            # for fedprox
            fed_prox_reg = 0.0
            # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))


    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc


def train_net_fedcon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, temperature, args,
                      round, device="cpu"):
    # net = nn.DataParallel(net)
    # net.cuda()
    net = to_device(net, device)
    logger.info('Training network %s' % str(net_id)) # 클라이언트 
    logger.info('n_training: %d' % len(train_dataloader)) # 배치
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)

    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))


    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)
    # global_net.to(device)

    for previous_net in previous_nets:
        previous_net.to(device)
    global_w = global_net.state_dict()

    cnt = 0
    cos=torch.nn.CosineSimilarity(dim=-1)
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _, pro1, out = net(x)
            _, pro2, _ = global_net(x)

            posi = cos(pro1, pro2)
            logits = posi.reshape(-1,1)

            for previous_net in previous_nets:
                previous_net.to(device)
                _, pro3, _ = previous_net(x)
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                previous_net.to('cpu')

            logits /= temperature
            labels = torch.zeros(x.size(0), device=device).long()

            loss2 = mu * criterion(logits, labels)


            loss1 = criterion(out, target)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))


    for previous_net in previous_nets:
        previous_net.to('cpu')
    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc


def local_train_net(nets, args, net_dataidx_map, train_dl=None, test_dl=None, global_model = None, prev_model_pool = None, server_c = None, clients_c = None, round=None, device="cpu"):
    avg_acc = 0.0
    acc_list = []
    if global_model:
        global_model.to(device)
    if server_c:
        server_c.to(device)
        server_c_collector = list(server_c.to(device).parameters())
        new_server_c_collector = copy.deepcopy(server_c_collector)
    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs))) # 클라이언트 번호, 데이터 수
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs) # 클라이언트별 데이터를 가져옴
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        if args.alg == 'fedavg':
            trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                        device=device)
        elif args.alg == 'fedprox':
            trainacc, testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args.mu, args, device=device)
        elif args.alg == 'moon':
            prev_models=[]
            for i in range(len(prev_model_pool)):
                prev_models.append(prev_model_pool[i][net_id])
            trainacc, testacc = train_net_fedcon(net_id, net, global_model, prev_models, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args.mu, args.temperature, args, round, device=device)

        elif args.alg == 'local_training':
            trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                          device=device)
        # fedbabu
        elif args.alg == 'fedbabu':
            trainacc, testacc = train_net_fedbabu(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args, device=device)
        # fedcap
        elif args.alg == 'fedcap':
            # prev_model_pool는 fedcap_hist (dict: cid -> deque[snapshots])로 전달됨
            previous_snaps = list(prev_model_pool[net_id]) if prev_model_pool is not None else []
            trainacc, testacc = train_net_fedcap(
                net_id, net, global_model, previous_snaps,train_dl_local, test_dl, n_epoch, args.lr,args.optimizer, args.mu, args.temperature, args, round, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        acc_list.append(testacc)
    avg_acc /= args.n_parties
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
        logger.info("std acc %f" % np.std(acc_list))
    if global_model:
        global_model.to('cpu')
    if server_c:
        for param_index, param in enumerate(server_c.parameters()):
            server_c_collector[param_index] = new_server_c_collector[param_index]
        server_c.to('cpu')
    return nets

# fedbabu 학습
def train_net_fedbabu(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    # net = nn.DataParallel(net)
    # net.cuda()
    net = to_device(net, device)
    logger.info('Training network %s (FedBABU)' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    # 1) head(l3) freeze (ModelFedCon 기준)
    base = unwrap_dp(net)
    if hasattr(base, "l3"):
        for p in base.l3.parameters():
            p.requires_grad = False
    else:
        logger.warning("FedBABU: net has no attribute l3. (Check model definition)")

    # 2) optimizer: requires_grad=True만 학습 (head 자동 제외)
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg, amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad()

            target = target.long()
            _, _, out = net(x)          # logits만
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()
            epoch_loss_collector.append(loss.item())

        logger.info('Epoch: %d Loss: %f' % (epoch, sum(epoch_loss_collector)/len(epoch_loss_collector)))
    logger.info(' ** FedBABU Training complete **')
    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, _, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    net.to("cpu")
    return train_acc, test_acc

# fedcap 학습
def train_net_fedcap(net_id, net, global_net, previous_snapshots,
                     train_dataloader, test_dataloader,
                     epochs, lr, args_optimizer, mu, temperature, args,
                     round, device="cpu"):

    # net = nn.DataParallel(net)
    # net.cuda()
    net = to_device(net, device)

    logger.info('Training network %s (FedCAP)' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    # --- FedBABU rule: freeze head (l3) ---
    for name, p in net.named_parameters():
        # DataParallel이면 name이 "module.l3.weight" 형태가 됨
        if "l3." in name:
            p.requires_grad = False

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg, amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)
    cos = torch.nn.CosineSimilarity(dim=-1)

    # global_net은 projection만 뽑는 용도
    global_net.to(device)
    global_net.eval()
    for p in global_net.parameters():
        p.requires_grad = False

    # 임시 모델: history snapshot 로드해서 pro3 계산
    # (global_net과 같은 구조의 모델을 복제)
    tmp_net = copy.deepcopy(global_net)
    tmp_net.to(device)
    tmp_net.eval()
    for p in tmp_net.parameters():
        p.requires_grad = False

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []

        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device).long()

            optimizer.zero_grad()

            # local forward
            _, pro1, out = net(x)           # pro1: local projection
            # global projection (positive)
            with torch.no_grad():
                _, pro2, _ = global_net(x)  # pro2: global projection

            posi = cos(pro1, pro2)
            logits = posi.reshape(-1, 1)

            # negatives: self-history snapshots
            if previous_snapshots is not None and len(previous_snapshots) > 0:
                for snap in previous_snapshots:
                    load_body_proj_state(tmp_net, snap)   # <-- 여기서 load_body_proj_state 사용
                    with torch.no_grad():
                        _, pro3, _ = tmp_net(x)           # pro3: history projection
                    nega = cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                logits /= temperature
                labels = torch.zeros(x.size(0), device=device).long()  # positive가 0번 컬럼
                loss2 = mu * criterion(logits, labels)
            else:
                # history가 없으면 contrastive 없음
                loss2 = torch.tensor(0.0, device=x.device)

            # supervised loss (head frozen이지만 gradient는 body+proj로 흐름)
            loss1 = criterion(out, target)

            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))
    
    logger.info(' ** FedCAP Training complete **')
    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, _, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    net.to("cpu")
    global_net.to("cpu")
    tmp_net.to("cpu")
    return train_acc, test_acc

if __name__ == '__main__':
    args, run_dir = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts = partition_data(
    args.dataset, args.datadir, args.logdir, args.partition, args.n_parties,
    beta=args.beta, holdout_ratio=0.2, seed=args.init_seed)

    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)


    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                               args.datadir,
                                                                               args.batch_size,
                                                                               32) # 전체 데이터셋 로딩

    print("len train_dl_global:", len(train_ds_global)) # 전체 데이터셋 크기
    train_dl=None
    data_size = len(test_ds_global)

    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device='cpu')

    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device='cpu')
    global_model = global_models[0]
    
    n_comm_rounds = args.comm_round

    start_round = 0
    old_nets_pool = None
    fedcap_hist = None

    if args.resume_ckpt is not None:
        ckpt = load_ckpt(args.resume_ckpt, map_location="cpu")
        start_round = ckpt["round"]

        # 복구: 참가자 목록
        party_list_rounds = ckpt["party_list_rounds"]

        # 복구: global model
        global_model.load_state_dict(ckpt["global_model"])

        # 복구: 이전 모델(moon, fedcap)
        old_nets_pool = ckpt.get("old_nets_pool", None)
        fedcap_hist = ckpt.get("fedcap_hist", None)

        # RNG 복구
        restore_rng_state(ckpt["rng_state"])

        # data map 복구
        if "net_dataidx_map_train" in ckpt:
            net_dataidx_map_train = ckpt["net_dataidx_map_train"]
            net_dataidx_map_test = ckpt["net_dataidx_map_test"]

    # ===== after global_model is defined =====
    # head 인지 여부
    def is_head_key(k: str) -> bool:
        return k.startswith("l3.") or k.startswith("module.l3.")
    # head 파라미터만 복사해서 반환
    def extract_head_sd(model):
        sd = model.state_dict()
        return {k: v.detach().clone() for k, v in sd.items() if is_head_key(k)}
    # 저장해둔 헤드만 다시 모델에 덮어쓰기
    def load_head_sd(model, head_sd):
        sd = model.state_dict()
        for k, v in head_sd.items():
            sd[k] = v
        model.load_state_dict(sd)
    # 고정된 헤드 글로벌 모델 + 클라이언트 모델에 똑같이 덮어씌움
    def broadcast_fixed_head(nets, global_model, fixed_head_sd):
        load_head_sd(global_model, fixed_head_sd) # 글로벌 모델에 헤드 덮어씌움
        for cid in nets.keys():
            load_head_sd(nets[cid], fixed_head_sd) # 클라이언트에게 헤드 덮어씌움
    # dict of list로 변환(저장하기 위해)
    def serialize_fedcap_hist(fedcap_hist):
        if fedcap_hist is None:
            return None
        return {cid: list(deq) for cid, deq in fedcap_hist.items()}
    # 원래 쓰던 구조로 복원
    def deserialize_fedcap_hist(obj, maxlen):
        if obj is None:
            return None
        dd = defaultdict(lambda: deque(maxlen=maxlen))
        for cid, snaps in obj.items():
            dd[int(cid)] = deque(snaps, maxlen=maxlen)
        return dd
        

    # create fixed shared head once
    fixed_head_sd = None
    if args.alg in ["fedbabu", "fedcap"]:
        fixed_head_sd = extract_head_sd(global_model)  # <- global head를 기준으로
        broadcast_fixed_head(nets, global_model, fixed_head_sd)
        logger.info("FedBABU/FedCAP: FIXED shared random head is broadcast to all clients.")
    # 바디만 로드
    def load_body_only(net, global_w):
        local_w = net.state_dict()
        for k in global_w:
            if k.startswith("l3.") or k.startswith("module.l3."):
                continue
            local_w[k] = global_w[k]
        net.load_state_dict(local_w)

    if args.resume_ckpt is None and args.load_model_file and args.alg != 'plot_visual':
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0

    if args.alg == 'moon':
        if old_nets_pool is None:
            old_nets_pool = []
        if args.load_pool_file:
            for nets_id in range(args.model_buffer_size):
                old_nets, _, _ = init_nets(args.net_config, args.n_parties, args, device='cpu')
                checkpoint = torch.load(args.load_pool_file)
                for net_id, net in old_nets.items():
                    net.load_state_dict(checkpoint['pool' + str(nets_id) + '_'+'net'+str(net_id)])
                old_nets_pool.append(old_nets)
        elif args.load_first_net:
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                old_nets_pool.append(old_nets)
        for round in range(start_round, n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()

            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)


            local_train_net(nets_this_round, args, net_dataidx_map_train, train_dl=train_dl, test_dl=test_dl, global_model = global_model, prev_model_pool=old_nets_pool, round=round, device=device)



            total_data_points = sum([len(net_dataidx_map_train[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in party_list_this_round]


            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)
            #summary(global_model.to(device), (3, 32, 32))

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.to(device)
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            global_model.to('cpu')
            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)


            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                old_nets_pool.append(old_nets)
            elif args.pool_option == 'FIFO':
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                for i in range(args.model_buffer_size-2, -1, -1):
                    old_nets_pool[i] = old_nets_pool[i+1]
                old_nets_pool[args.model_buffer_size - 1] = old_nets

            mkdirs(args.modeldir+'fedcon/')
            if args.save_model:
                torch.save(global_model.state_dict(), args.modeldir+'fedcon/global_model_'+args.log_file_name+'.pth')
                torch.save(nets[0].state_dict(), args.modeldir+'fedcon/localmodel0'+args.log_file_name+'.pth')
                for nets_id, old_nets in enumerate(old_nets_pool):
                    torch.save({'pool'+ str(nets_id) + '_'+'net'+str(net_id): net.state_dict() for net_id, net in old_nets.items()}, args.modeldir+'fedcon/prev_model_pool_'+args.log_file_name+'.pth')
            
            if (round + 1) % args.ckpt_every == 0:
                ckpt_payload = {
                    "round": round + 1,  # 다음 시작 라운드
                    "global_model": global_model.state_dict(),
                    "party_list_rounds": party_list_rounds,
                    "rng_state": capture_rng_state(),
                    "old_nets_pool": old_nets_pool,
                    "fedcap_hist": serialize_fedcap_hist(fedcap_hist),                    
                    "net_dataidx_map_train": net_dataidx_map_train,
                    "net_dataidx_map_test": net_dataidx_map_test,
                }
                save_ckpt(os.path.join(run_dir, "ckpt", f"ckpt_round{round+1}.pth"), ckpt_payload)
                save_ckpt(os.path.join(run_dir, "ckpt", "latest.pth"), ckpt_payload)
        
    elif args.alg == 'fedavg':
        for round in range(start_round, n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, net_dataidx_map_train, train_dl=train_dl, test_dl=test_dl, device=device)

            total_data_points = sum([len(net_dataidx_map_train[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]


            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]


            global_model.load_state_dict(global_w)

            #logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)
            mkdirs(args.modeldir+'fedavg/')
            global_model.to('cpu')

            torch.save(global_model.state_dict(), args.modeldir+'fedavg/'+'globalmodel'+args.log_file_name+'.pth')
            torch.save(nets[0].state_dict(), args.modeldir+'fedavg/'+'localmodel0'+args.log_file_name+'.pth')

            if (round + 1) % args.ckpt_every == 0:
                ckpt_payload = {
                    "round": round + 1,  # 다음 시작 라운드
                    "global_model": global_model.state_dict(),
                    "party_list_rounds": party_list_rounds,
                    "rng_state": capture_rng_state(),
                    "old_nets_pool": old_nets_pool,
                    "fedcap_hist": serialize_fedcap_hist(fedcap_hist),                    
                    "net_dataidx_map_train": net_dataidx_map_train,
                    "net_dataidx_map_test": net_dataidx_map_test,
                }
                save_ckpt(os.path.join(run_dir, "ckpt", f"ckpt_round{round+1}.pth"), ckpt_payload)
                save_ckpt(os.path.join(run_dir, "ckpt", "latest.pth"), ckpt_payload)
    
    elif args.alg == 'fedprox':

        for round in range(start_round, n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]
            global_w = global_model.state_dict()
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)


            local_train_net(nets_this_round, args, net_dataidx_map_train, train_dl=train_dl,test_dl=test_dl, global_model = global_model, device=device)
            global_model.to('cpu')

            # update global model
            total_data_points = sum([len(net_dataidx_map_train[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
            global_model.load_state_dict(global_w)


            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))

            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)
            mkdirs(args.modeldir + 'fedprox/')
            global_model.to('cpu')
            torch.save(global_model.state_dict(), args.modeldir +'fedprox/'+args.log_file_name+ '.pth')

            if (round + 1) % args.ckpt_every == 0:
                ckpt_payload = {
                    "round": round + 1,  # 다음 시작 라운드
                    "global_model": global_model.state_dict(),
                    "party_list_rounds": party_list_rounds,
                    "rng_state": capture_rng_state(),
                    "old_nets_pool": old_nets_pool,
                    "fedcap_hist": serialize_fedcap_hist(fedcap_hist),
                    "net_dataidx_map_train": net_dataidx_map_train,
                    "net_dataidx_map_test": net_dataidx_map_test,
                }
                save_ckpt(os.path.join(run_dir, "ckpt", f"ckpt_round{round+1}.pth"), ckpt_payload)
                save_ckpt(os.path.join(run_dir, "ckpt", "latest.pth"), ckpt_payload)

    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        local_train_net(nets, args, net_dataidx_map_train, train_dl=train_dl,test_dl=test_dl, device=device)
        mkdirs(args.modeldir + 'localmodel/')
        for net_id, net in nets.items():
            torch.save(net.state_dict(), args.modeldir + 'localmodel/'+'model'+str(net_id)+args.log_file_name+ '.pth')

    elif args.alg == 'all_in':
        nets, _, _ = init_nets(args.net_config, 1, args, device='cpu')
        # nets[0].to(device)
        trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl, args.epochs, args.lr,
                                      args.optimizer, args, device=device)
        logger.info("All in test acc: %f" % testacc)
        mkdirs(args.modeldir + 'all_in/')

        torch.save(nets[0].state_dict(), args.modeldir+'all_in/'+args.log_file_name+ '.pth')
    # fedbabu
    elif args.alg == 'fedbabu':

        for round in range(start_round, n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                load_body_only(net, global_w)

            local_train_net(nets_this_round, args, net_dataidx_map_train,
                            train_dl=train_dl, test_dl=test_dl, device=device)

            total_data_points = sum([len(net_dataidx_map_train[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()

                if net_id == 0:
                    for key in net_para:
                        if key.startswith("l3.") or key.startswith("module.l3."):
                            continue
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        if key.startswith("l3.") or key.startswith("module.l3."):
                            continue
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            global_model.load_state_dict(global_w)

            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            global_model.to('cpu')

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

            if (round + 1) % args.ckpt_every == 0:
                ckpt_payload = {
                    "round": round + 1,  # 다음 시작 라운드
                    "global_model": global_model.state_dict(),
                    "party_list_rounds": party_list_rounds,
                    "rng_state": capture_rng_state(),
                    "old_nets_pool": old_nets_pool,
                    "fedcap_hist": serialize_fedcap_hist(fedcap_hist),
                    "net_dataidx_map_train": net_dataidx_map_train,
                    "net_dataidx_map_test": net_dataidx_map_test,
                }
                save_ckpt(os.path.join(run_dir, "ckpt", f"ckpt_round{round+1}.pth"), ckpt_payload)
                save_ckpt(os.path.join(run_dir, "ckpt", "latest.pth"), ckpt_payload)
    # fedcap
    elif args.alg == 'fedcap':
        if fedcap_hist is None:
            fedcap_hist = defaultdict(lambda: deque(maxlen=args.model_buffer_size))
        else:
            # 혹시 plain dict로 들어왔으면 defaultdict로 복구
            if not isinstance(fedcap_hist, defaultdict):
                fedcap_hist = deserialize_fedcap_hist(fedcap_hist, args.model_buffer_size)
        
        # (옵션) load_first_net이면 첫 라운드 history 초기화(비어있어도 상관없어서 없어도 됨)
        # 보통은 비워두고 시작해도 OK

        for round in range(start_round, n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()

            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}

            # --- broadcast: body+proj만 (head 유지) ---
            for net in nets_this_round.values():
                load_body_proj_only(net, global_model)   # 모델을 넘김

            # --- local update (FedCAP): prev_model_pool 자리에 fedcap_hist를 넘김 ---
            local_train_net(
                nets_this_round, args, net_dataidx_map_train,
                train_dl=train_dl, test_dl=test_dl,
                global_model=global_model,
                prev_model_pool=fedcap_hist,      # <-- 여기서 fedcap_hist 사용
                round=round,
                device=device
            )

            # --- aggregation: body+proj만 평균 (l3 제외) ---
            total_data_points = sum([len(net_dataidx_map_train[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        if key.startswith("l3."):
                            continue
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        if key.startswith("l3."):
                            continue
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    if key.startswith("l3."):
                        continue
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w, strict=False)

            # --- global eval (moon이랑 동일) ---
            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            global_model.to('cpu')
            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)

            # --- history update: 참여한 cid만 snapshot 저장 (θ,ψ만) ---
            for cid in party_list_this_round:
                fedcap_hist[cid].append(extract_body_proj_state(nets[cid]))  # <-- 여기서 extract_body_proj_state 사용

            if (round + 1) % args.ckpt_every == 0:
                ckpt_payload = {
                    "round": round + 1,  # 다음 시작 라운드
                    "global_model": global_model.state_dict(),
                    "party_list_rounds": party_list_rounds,
                    "rng_state": capture_rng_state(),
                    "old_nets_pool": old_nets_pool,
                    "net_dataidx_map_train": net_dataidx_map_train,
                    "net_dataidx_map_test": net_dataidx_map_test,
                    "fedcap_hist": serialize_fedcap_hist(fedcap_hist),
                }
                save_ckpt(os.path.join(run_dir, "ckpt", f"ckpt_round{round+1}.pth"), ckpt_payload)
                save_ckpt(os.path.join(run_dir, "ckpt", "latest.pth"), ckpt_payload)

    # ---- evaluation ----
    eval_clients = list(nets.keys())

    g_acc = evaluate_generalization_head_avg(
        global_model=global_model,
        client_list=eval_clients,
        nets=nets,
        net_dataidx_map=net_dataidx_map_train,
        get_dataloader_fn=get_dataloader,
        compute_accuracy_fn=compute_accuracy,
        test_dl=test_dl,
        args=args,
        Kg=args.Kg,
        head_lr=args.head_lr,
        device=device
    )
    logger.info(">> Generalization acc (head-avg): %f" % g_acc)

    p_acc_mean, _ = evaluate_personalization(
        global_model=global_model,
        client_list=eval_clients,
        nets=nets,
        net_dataidx_map_train=net_dataidx_map_train,
        net_dataidx_map_test=net_dataidx_map_test,
        get_dataloader_fn=get_dataloader,
        compute_accuracy_fn=compute_accuracy,
        test_dl=test_dl,
        args=args,
        Kp=args.Kp,
        head_lr=args.head_lr,
        device=device
    )
    save_experiment(run_dir, args, global_model, g_acc, p_acc_mean)
    print("Experiment saved.")

    logger.info(">> Personalization acc (mean): %f" % p_acc_mean)

    logger.info("[FINAL] alg=%s Kg=%d Kp=%d GenAcc=%.4f PersAcc=%.4f" %
                (args.alg, args.Kg, args.Kp, g_acc, p_acc_mean))