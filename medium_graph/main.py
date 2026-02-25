import argparse
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from collections import OrderedDict
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
import wandb

from logger import *
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, load_fixed_splits, class_rand_splits
from eval import *
from parse import parse_method, parser_add_main_args

def _entropy(p, q):
    if abs(q - 1.0) > 1e-6:
        return (1.0 / (q - 1.0)) * (1.0 - (p ** q).sum(1))
    return -(p * p.log()).sum(1)

class LambdaMetaNet(nn.Module):
    def __init__(self, in_dim=7, hidden=32, lambda_max=1.0):
        super().__init__()
        self.lambda_max = lambda_max
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def forward(self, x):
        return self.lambda_max * torch.sigmoid(self.net(x)).squeeze()


def split_train_for_meta_v3(train_idx, holdout_frac, seed, run_idx, epoch, device):
    num_train = train_idx.numel()
    if num_train < 2:
        raise ValueError("Need at least 2 training nodes for --learn_lambda_meta_v3.")

    num_meta = int(round(holdout_frac * num_train))
    num_meta = max(1, min(num_meta, num_train - 1))

    g = torch.Generator(device="cpu")
    g.manual_seed(
        int(seed)
        + 100000 * int(run_idx)
        + 1000 * int(epoch)
    )

    perm = torch.randperm(num_train, generator=g).to(device)

    meta_train_idx = train_idx[perm[:num_meta]]
    inner_train_idx = train_idx[perm[num_meta:]]

    return inner_train_idx, meta_train_idx

def ts_loss_from_logits(
    logits,
    labels,
    train_idx,
    train_mask,
    val_mask,
    test_mask,
    criterion,
    q,
    scale,
    lambda_u,
    lambda_val,
    lambda_t,
    mi_max,
):
    log_probs = F.log_softmax(logits, dim=1)
    ce_loss = criterion(log_probs[train_idx], labels.squeeze(1)[train_idx])
    loss = ce_loss

    p_test = F.softmax(logits[test_mask], dim=1)
    p_val = F.softmax(logits[val_mask], dim=1)
    p_train = F.softmax(logits[train_mask], dim=1)

    ent_test = _entropy(p_test, q)
    ent_val = _entropy(p_val, q)
    ent_train = _entropy(p_train, q)

    p_mean = torch.cat([p_test, p_val]).mean(dim=0)
    if abs(q - 1.0) > 1e-6:
        h_marginal = (1.0 / (q - 1.0)) * (1.0 - (p_mean ** q).sum())
    else:
        h_marginal = -(p_mean * (p_mean + 1e-10).log()).sum()

    n_unlabeled = ent_test.shape[0] + ent_val.shape[0]

    loss = loss + scale * lambda_u * ent_test.sum() / n_unlabeled
    loss = loss + scale * lambda_val * ent_val.sum() / n_unlabeled
    loss = loss - scale * mi_max * h_marginal
    loss = loss + scale * lambda_t * ent_train.mean()

    stats = {
        "ce_loss": ce_loss,
        "entropy_test": ent_test.mean(),
        "entropy_val": ent_val.mean(),
        "entropy_labeled": ent_train.mean(),
        "entropy_unlabeled": torch.cat([ent_test, ent_val]).mean(),
        "entropy_marginal": h_marginal,
    }

    return loss, stats

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

### Parse args ###
parser = argparse.ArgumentParser(description='Training Pipeline for Node Classification')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

lambda_meta_modes = [
    args.learn_lambda_meta,
    args.learn_lambda_meta_v2,
    args.learn_lambda_meta_v3,
]
if sum(bool(x) for x in lambda_meta_modes) > 1:
    raise ValueError(
        "Use only one of --learn_lambda_meta, --learn_lambda_meta_v2, "
        "or --learn_lambda_meta_v3 at a time."
    )

if args.learn_lambda_meta_v3:
    if not (0.0 < args.meta_train_holdout_frac < 1.0):
        raise ValueError("--meta_train_holdout_frac must be in (0, 1).")

### Tie lambda_t to -lambda_u if specified ###
if args.tie_lambda_t:
    args.lambda_t = -args.lambda_u
args.lambda_u += args.midpoint_offset
args.lambda_t += args.midpoint_offset
if args.lambda_val is None:
    args.lambda_val = args.lambda_u

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)

if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [class_rand_splits(
        dataset.label, args.label_num_per_class, args.valid_num, args.test_num)]
else:
    split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)


dataset.label = dataset.label.to(device)

### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

### Load method ###
model = parse_method(args, n, c, d, device)

use_lambda_meta_net = (
    args.learn_lambda_meta
    or args.learn_lambda_meta_v2
    or args.learn_lambda_meta_v3
)

if use_lambda_meta_net:
    meta_net = LambdaMetaNet(
        in_dim=7,
        hidden=args.meta_hidden,
        lambda_max=args.lambda_max,
    ).to(device)
else:
    meta_net = None

### Loss function (Single-class, Multi-class) ###
if args.dataset in ('questions'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

### Performance metric (Acc, AUC) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
else:
    eval_func = eval_acc

args.method = args.gnn
logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

run = wandb.init(
    project="transductive-sharpening",
    group=f"{args.gnn}_{args.dataset}",
    name=f"{args.gnn}_{args.dataset}",
    reinit=True,
)
run.config.update(vars(args), allow_val_change=True)
run.tags += ("tg",)

### Training loop ###
all_best_vals = []
all_test_at_best_vals = []
for run in range(args.runs):
    if args.dataset in ('coauthor-cs', 'coauthor-physics', 'amazon-computer', 'amazon-photo', 'cora', 'citeseer', 'pubmed'):
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    train_mask = torch.zeros(n, dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    val_idx = split_idx['valid'].to(device)
    val_mask = torch.zeros(n, dtype=torch.bool, device=device)
    val_mask[val_idx] = True
    # used only for label-free training; transductive setting permits this
    test_mask = ~train_mask & ~val_mask
    inner_train_idx = train_idx
    meta_train_idx = None
    inner_train_mask = train_mask
    meta_train_mask = None

    loss_train_idx = train_idx
    loss_train_mask = train_mask
    meta_objective_idx = val_idx
    loss_val_mask = val_mask
    model.reset_parameters()
    if use_lambda_meta_net:
        meta_net.reset_parameters()
    if args.learn_lambda_meta_v2:
        # v2: lambda meta-network is learned jointly with the model from the same loss.
        # Use param groups so the model and meta-network can keep separate learning rates.
        optimizer = torch.optim.Adam(
            [
                {
                    "params": model.parameters(),
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": meta_net.parameters(),
                    "lr": args.meta_lr,
                    "weight_decay": 0.0,
                },
            ]
        )
        meta_optimizer = None
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            weight_decay=args.weight_decay,
            lr=args.lr,
        )
        if args.learn_lambda_meta or args.learn_lambda_meta_v3:
            meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=args.meta_lr)
        else:
            meta_optimizer = None
    best_val = float('-inf')
    test_at_best_val = float('-inf')
    if args.save_model:
        save_model(args, model, optimizer, run)

    for epoch in range(args.epochs):
        if args.learn_lambda_meta_v3:
            inner_train_idx, meta_train_idx = split_train_for_meta_v3(
                train_idx=train_idx,
                holdout_frac=args.meta_train_holdout_frac,
                seed=args.seed,
                run_idx=run,
                epoch=epoch,
                device=device,
            )

            inner_train_mask = torch.zeros(n, dtype=torch.bool, device=device)
            inner_train_mask[inner_train_idx] = True

            meta_train_mask = torch.zeros(n, dtype=torch.bool, device=device)
            meta_train_mask[meta_train_idx] = True

            loss_train_idx = inner_train_idx
            loss_train_mask = inner_train_mask
            meta_objective_idx = meta_train_idx
            loss_val_mask = val_mask | meta_train_mask

        model.train()
        optimizer.zero_grad()

        out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
        if args.dataset in ('questions'):
            if dataset.label.shape[1] == 1:
                true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
            else:
                true_label = dataset.label
            loss = criterion(out[train_idx], true_label.squeeze(1)[
                train_idx].to(torch.float))
        else:
            logits = out
            scale = min(1.0, epoch / (args.warmup_frac * args.epochs + 1e-8))
            q = args.tsallis_q

            fixed_lambda_u = torch.tensor(float(args.lambda_u), device=device)
            fixed_lambda_val = torch.tensor(float(args.lambda_val), device=device)
            fixed_lambda_t = torch.tensor(float(args.lambda_t), device=device)

            base_loss, base_stats = ts_loss_from_logits(
                logits=logits,
                labels=dataset.label,
                train_idx=train_idx,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                criterion=criterion,
                q=q,
                scale=scale,
                lambda_u=fixed_lambda_u,
                lambda_val=fixed_lambda_val,
                lambda_t=fixed_lambda_t,
                mi_max=args.mi_max,
            )

            meta_val_loss = None

            lambda_mode_active = (
                epoch >= args.meta_warmup_epochs
                and use_lambda_meta_net
            )

            if lambda_mode_active:
                meta_features = torch.stack([
                    torch.tensor(epoch / args.epochs, device=device),
                    base_stats["ce_loss"].detach(),
                    base_stats["entropy_labeled"].detach(),
                    base_stats["entropy_val"].detach(),
                    # label-free statistic; transductive setting permits this
                    base_stats["entropy_test"].detach(),
                    base_stats["entropy_unlabeled"].detach(),
                    base_stats["entropy_marginal"].detach(),
                ]).float()

            if args.learn_lambda_meta_v2 and lambda_mode_active:
                lambda_used = meta_net(meta_features)
                lambda_val_used = lambda_used
                lambda_t_used = -lambda_used

            elif (args.learn_lambda_meta or args.learn_lambda_meta_v3) and lambda_mode_active:
                lambda_meta = meta_net(meta_features)

                train_loss_meta, _ = ts_loss_from_logits(
                    logits=logits,
                    labels=dataset.label,
                    train_idx=loss_train_idx,
                    train_mask=loss_train_mask,
                    val_mask=loss_val_mask,
                    test_mask=test_mask,
                    criterion=criterion,
                    q=q,
                    scale=scale,
                    lambda_u=lambda_meta,
                    lambda_val=lambda_meta,
                    lambda_t=-lambda_meta,
                    mi_max=args.mi_max,
                )

                params = OrderedDict(model.named_parameters())
                grads = torch.autograd.grad(
                    train_loss_meta,
                    params.values(),
                    create_graph=True,
                    allow_unused=True,
                )

                fast_params = OrderedDict()
                for (name, param), grad in zip(params.items(), grads):
                    if grad is None:
                        fast_params[name] = param
                    else:
                        fast_params[name] = param - args.lr * grad

                val_logits = functional_call(
                    model,
                    fast_params,
                    (dataset.graph['node_feat'], dataset.graph['edge_index']),
                )

                val_log_probs = F.log_softmax(val_logits, dim=1)
                meta_val_loss = criterion(
                    val_log_probs[meta_objective_idx],
                    dataset.label.squeeze(1)[meta_objective_idx],
                )

                meta_optimizer.zero_grad()
                meta_val_loss.backward()
                meta_optimizer.step()

                logits = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

                with torch.no_grad():
                    lambda_used = meta_net(meta_features).detach()

                lambda_val_used = lambda_used
                lambda_t_used = -lambda_used

            else:
                lambda_used = fixed_lambda_u
                lambda_val_used = fixed_lambda_val
                lambda_t_used = fixed_lambda_t

            loss, ts_stats = ts_loss_from_logits(
                logits=logits,
                labels=dataset.label,
                train_idx=loss_train_idx,
                train_mask=loss_train_mask,
                val_mask=loss_val_mask,
                test_mask=test_mask,
                criterion=criterion,
                q=q,
                scale=scale,
                lambda_u=lambda_used,
                lambda_val=lambda_val_used,
                lambda_t=lambda_t_used,
                mi_max=args.mi_max,
            )

            ce_loss = ts_stats["ce_loss"]
            ent_test = ts_stats["entropy_test"]
            ent_val = ts_stats["entropy_val"]
            ent_train = ts_stats["entropy_labeled"]
            h_marginal = ts_stats["entropy_marginal"]

        loss.backward()
        optimizer.step()

        result = evaluate(model, dataset, split_idx, eval_func, criterion, args)

        logger.add_result(run, result[:-1])

        if result[1] > best_val:
            best_val = result[1]
            test_at_best_val = result[2]
            if args.save_model:
                save_model(args, model, optimizer, run)

        metrics = {
            "epoch": epoch,
            "run_idx": run,
            "loss": loss.item(),
            "train_acc": result[0],
            "val_acc": result[1],
            "test_acc": result[2],
            "best_val": best_val,
            "test_at_best_val": test_at_best_val,
        }
        if args.dataset not in ('questions'):
            metrics.update({
                "ce_loss": ce_loss.item(),
                "entropy_unlabeled": ts_stats["entropy_unlabeled"].item(),
                "entropy_test": ent_test.item(),
                "entropy_val": ent_val.item(),
                "entropy_labeled": ent_train.item(),
                "entropy_marginal": h_marginal.item(),
                "lambda_used": float(lambda_used.item()),
            })

            if meta_val_loss is not None:
                metrics.update({
                    "meta_val_loss": float(meta_val_loss.item()),
                })
        wandb.log(metrics)

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%, '
                  f'Best Valid: {100 * best_val:.2f}%, '
                  f'Best Test: {100 * test_at_best_val:.2f}%')
    logger.print_statistics(run)
    all_best_vals.append(best_val)
    all_test_at_best_vals.append(test_at_best_val)

val_mean = float(np.mean(all_best_vals))
val_std = float(np.std(all_best_vals, ddof=1))
test_mean = float(np.mean(all_test_at_best_vals))
test_std = float(np.std(all_test_at_best_vals, ddof=1))

wandb.log({
    "val_mean": val_mean,
    "val_std": val_std,
    "test_mean": test_mean,
    "test_std": test_std,
})

wandb.summary["val_mean"] = val_mean
wandb.summary["val_std"] = val_std
wandb.summary["test_mean"] = test_mean
wandb.summary["test_std"] = test_std

wandb.finish()

results = logger.print_statistics()
### Save results ###
save_result(args, results)

