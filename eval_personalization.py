# eval_personalization.py
import copy
import logging
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)

# -----------------------------
# Head 관련 유틸
# -----------------------------
def is_head_key(k: str) -> bool:
    return k.startswith("l3.") or k.startswith("module.l3.")

def overwrite_from_global(local_model, global_sd):
    local_model.load_state_dict(global_sd, strict=True)

def freeze_body_train_head(model):
    """
    head(l3)만 학습 가능, 나머지(body+proj)는 freeze
    """
    head_cnt, body_cnt = 0, 0
    for name, p in model.named_parameters():
        if name.startswith("l3") or name.startswith("module.l3"):
            p.requires_grad = True
            head_cnt += p.numel()
        else:
            p.requires_grad = False
            body_cnt += p.numel()

    logger.info(f"[freeze] trainable(head) params={head_cnt:,} | frozen(body+proj) params={body_cnt:,}")

def _head_norm(model) -> float:
    sd = model.state_dict()
    keys = [k for k in sd.keys() if is_head_key(k)]
    if not keys:
        return -1.0
    tot = 0.0
    for k in keys:
        tot += sd[k].float().norm().item()
    return tot

def freeze_bn_stats(model):
    """
    BatchNorm running stats 업데이트 방지.
    requires_grad와 무관하게 train() 상태에서 BN이 흔들리는 문제를 막음.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()

def finetune_head_steps(model, train_loader, K, lr, device="cuda"):
    model.to(device)
    model.train()

    # BN running stats 고정
    freeze_bn_stats(model)

    # head key sanity check
    sd_keys = list(model.state_dict().keys())
    head_keys = [k for k in sd_keys if is_head_key(k)]
    if len(head_keys) == 0:
        logger.warning("[finetune] No head keys matched by is_head_key(). "
                       "Your head may not be 'l3'. Check model.state_dict() keys!")

    freeze_body_train_head(model)

    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        logger.warning("[finetune] No trainable params found! Head FT will do nothing.")
        model.to("cpu")
        return

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
# Personalization Evaluation (client별 test로 평가)
#  - 전제: get_dataloader_fn이 (train_dl, test_dl, _, _) 형태로 리턴하며,
#         test_dl은 "client별 test"를 만들 수 있어야 함.
#  - 가장 깔끔한 건 net_dataidx_map_test를 따로 받아서 test_dataidxs로 넘기는 방식.
# -----------------------------
def evaluate_personalization(
    global_model,
    client_list,
    nets,
    net_dataidx_map_train,      # client별 train indices
    net_dataidx_map_test=None,  # client별 test indices (없으면 fallback)
    get_dataloader_fn=None,
    compute_accuracy_fn=None,
    test_dl=None,               # fallback용 global test loader
    args=None,
    Kp=15,
    head_lr=0.01,
    device="cuda"
):
    global_sd = copy.deepcopy(global_model.state_dict())
    accs = []

    logger.info(f"[PersEval] clients={len(client_list)} Kp={Kp} head_lr={head_lr} device={device}")

    for cid in client_list:
        local = copy.deepcopy(nets[cid])

        n_train = len(net_dataidx_map_train[cid])
        n_test_local = len(net_dataidx_map_test[cid]) if (net_dataidx_map_test is not None and cid in net_dataidx_map_test) else -1
        logger.info(f"[PersEval] cid={cid} n_train_samples={n_train} n_test_samples={n_test_local}")

        # 1) global 전체 로드
        overwrite_from_global(local, global_sd)

        # 2) 로컬 train/test loader 둘 다 받기
        #    - get_dataloader_fn이 client별 test를 만들려면 "test_dataidxs" 같은 인자를 받아야 함.
        #    - 너의 현재 get_dataloader()는 test_ds를 통째로 쓰는 구조라면, 아래의 test_dataidxs는 무시될 수 있음.
        if net_dataidx_map_test is not None and cid in net_dataidx_map_test:
            train_dl_local, test_dl_local, _, _ = get_dataloader_fn(
                args.dataset, args.datadir,
                args.batch_size, 32,
                dataidxs=net_dataidx_map_train[cid],
                test_dataidxs=net_dataidx_map_test[cid],   # client test
            )
        else:
            train_dl_local, test_dl_local, _, _ = get_dataloader_fn(
                args.dataset, args.datadir,
                args.batch_size, 32,
                dataidxs=net_dataidx_map_train[cid]
            )

        # 3) (선택) 로컬 test loader가 None이거나 비어있으면 fallback
        if test_dl_local is None:
            logger.warning(f"[PersEval] cid={cid} test_dl_local is None. Falling back to provided test_dl.")
            test_dl_local = test_dl

        # 4) FT 전: "클라이언트별 test"로 평가
        local.to(device)
        pre_acc, _, _ = compute_accuracy_fn(local, test_dl_local, get_confusion_matrix=True, device=device)
        pre_norm = _head_norm(local)
        local.to("cpu")
        logger.info(f"[PersEval] cid={cid} pre_acc={pre_acc:.4f} head_norm(pre)={pre_norm:.4f}")

        # 5) head FT (client train)
        finetune_head_steps(local, train_dl_local, K=Kp, lr=head_lr, device=device)

        # 6) FT 후: "클라이언트별 test"로 평가
        local.to(device)
        post_acc, _, _ = compute_accuracy_fn(local, test_dl_local, get_confusion_matrix=True, device=device)
        post_norm = _head_norm(local)
        local.to("cpu")
        logger.info(
            f"[PersEval] cid={cid} post_acc={post_acc:.4f} head_norm(post)={post_norm:.4f} "
            f"delta_norm={post_norm - pre_norm:.4f}"
        )

        accs.append(post_acc)

    mean_acc = sum(accs) / len(accs) if len(accs) else 0.0
    if len(accs) > 0:
        logger.info(f"[PersEval] mean={mean_acc:.4f} min={min(accs):.4f} max={max(accs):.4f}")
    else:
        logger.info(f"[PersEval] mean={mean_acc:.4f} (no clients)")
    return mean_acc, accs


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
    Kg,
    head_lr,
    device="cuda"
):
    global_sd = copy.deepcopy(global_model.state_dict())
    head_sds = []
    weights = []
    total = sum(len(net_dataidx_map[c]) for c in client_list)

    logger.info(f"[GenEval] clients={len(client_list)} Kg={Kg} head_lr={head_lr} device={device} total_train={total}")

    for cid in client_list:
        local = copy.deepcopy(nets[cid])
        overwrite_from_global(local, global_sd)

        train_dl_local, _, _, _ = get_dataloader_fn(
            args.dataset, args.datadir, args.batch_size, 32, net_dataidx_map[cid]
        )

        pre_norm = _head_norm(local)
        finetune_head_steps(local, train_dl_local, K=Kg, lr=head_lr, device=device)
        post_norm = _head_norm(local)

        head_sds.append(extract_head_sd(local))
        weights.append(len(net_dataidx_map[cid]) / total)

        logger.info(f"[GenEval] cid={cid} weight={weights[-1]:.4f} head_norm(pre)={pre_norm:.4f} "
                    f"head_norm(post)={post_norm:.4f} delta_norm={post_norm - pre_norm:.4f}")

    if len(head_sds) == 0:
        logger.warning("[GenEval] No heads collected. Returning 0.")
        return 0.0

    avg_head = average_head_sds(head_sds, weights)

    eval_model = copy.deepcopy(global_model)
    load_head_sd(eval_model, avg_head)

    eval_model.to(device)
    test_acc, _, _ = compute_accuracy_fn(eval_model, test_dl, get_confusion_matrix=True, device=device)
    eval_model.to("cpu")

    logger.info(f"[GenEval] test_acc={test_acc:.4f}")
    return test_acc