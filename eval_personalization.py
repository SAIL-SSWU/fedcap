# eval_personalization.py
import copy
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# Head 관련 유틸
# -----------------------------
def is_head_key(k: str) -> bool:
    return k.startswith("l3.") or k.startswith("module.l3.")


def overwrite_from_global(local_model, global_sd):
    """
    모든 알고리즘 공통:
    평가 시작 전, local 모델을 최종 global 모델 전체(바디+헤드)로 초기화
    """
    local_model.load_state_dict(global_sd, strict=True)


def freeze_body_train_head(model):
    """
    head(l3)만 학습 가능, 나머지(body+proj)는 freeze
    """
    for name, p in model.named_parameters():
        if name.startswith("l3") or name.startswith("module.l3"):
            p.requires_grad = True
        else:
            p.requires_grad = False


def finetune_head_steps(model, train_loader, K, lr, device="cuda"):
    model.to(device)
    model.train()

    freeze_body_train_head(model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss().to(device)

    step = 0
    while step < K:
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).long()

            optimizer.zero_grad()
            _, _, logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            step += 1
            if step >= K:
                break

    model.to("cpu")


def extract_head_sd(model):
    sd = model.state_dict()
    return {k: v.detach().clone() for k, v in sd.items() if is_head_key(k)}


def load_head_sd(model, head_sd):
    sd = model.state_dict()
    for k, v in head_sd.items():
        sd[k] = v
    model.load_state_dict(sd)


def average_head_sds(head_sds, weights):
    avg = {}
    for k in head_sds[0].keys():
        avg[k] = head_sds[0][k] * weights[0]
        for i in range(1, len(head_sds)):
            avg[k] += head_sds[i][k] * weights[i]
    return avg


# -----------------------------
# Personalization Evaluation
# -----------------------------
def evaluate_personalization(
    global_model,
    client_list,
    nets,
    net_dataidx_map,
    get_dataloader_fn,
    compute_accuracy_fn,
    test_dl,
    args,
    Kp=50,
    head_lr=0.01,
    device="cuda"
):
    """
    Personalization:
    1) 각 클라가 최종 global 모델 전체(바디+헤드)를 받음
    2) head만 Kp steps fine-tune
    3) 각 클라 test accuracy 평균
    """
    global_sd = copy.deepcopy(global_model.state_dict())
    accs = []

    for cid in client_list:
        local = copy.deepcopy(nets[cid])

        # 항상 global 전체 로드
        overwrite_from_global(local, global_sd)

        train_dl_local, _, _, _ = get_dataloader_fn(
            args.dataset, args.datadir, args.batch_size, 32, net_dataidx_map[cid]
        )

        finetune_head_steps(local, train_dl_local, K=Kp, lr=head_lr, device=device)

        local.to(device)
        test_acc, _, _ = compute_accuracy_fn(
            local, test_dl, get_confusion_matrix=True, device=device
        )
        local.to("cpu")
        accs.append(test_acc)

    return sum(accs) / len(accs), accs


# -----------------------------
# Generalization Evaluation
# -----------------------------
def evaluate_generalization_head_avg(
    global_model,
    client_list,
    nets,
    net_dataidx_map,
    get_dataloader_fn,
    compute_accuracy_fn,
    test_dl,
    args,
    Kg=50,
    head_lr=0.01,
    device="cuda"
):
    """
    Generalization (head-avg):
    1) 각 클라가 global 전체 모델 받음
    2) head만 Kg steps fine-tune
    3) 서버에서 fine-tuned head 평균
    4) (global body + averaged head)로 test
    """
    global_sd = copy.deepcopy(global_model.state_dict())

    head_sds = []
    weights = []
    total = sum(len(net_dataidx_map[c]) for c in client_list)

    for cid in client_list:
        local = copy.deepcopy(nets[cid])

        # 항상 global 전체 로드
        overwrite_from_global(local, global_sd)

        train_dl_local, _, _, _ = get_dataloader_fn(
            args.dataset, args.datadir, args.batch_size, 32, net_dataidx_map[cid]
        )

        finetune_head_steps(local, train_dl_local, K=Kg, lr=head_lr, device=device)

        head_sds.append(extract_head_sd(local))
        weights.append(len(net_dataidx_map[cid]) / total)

    avg_head = average_head_sds(head_sds, weights)

    # global body + averaged head
    eval_model = copy.deepcopy(global_model)
    load_head_sd(eval_model, avg_head)

    eval_model.to(device)
    test_acc, _, _ = compute_accuracy_fn(
        eval_model, test_dl, get_confusion_matrix=True, device=device
    )
    eval_model.to("cpu")

    return test_acc