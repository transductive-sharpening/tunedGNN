from pathlib import Path

from model import MPNNs, MLP

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"

def parse_method(args, n, c, d, device):
    if args.gnn == 'mlp':
        model = MLP(d, args.hidden_channels, c, local_layers=args.local_layers,
                     dropout=args.dropout, ln=args.ln, bn=args.bn,
                     res=args.res).to(device)
    else:
        model = MPNNs(d, args.hidden_channels, c, local_layers=args.local_layers,
                       dropout=args.dropout, heads=args.num_heads, pre_ln=args.pre_ln,
                       pre_linear=args.pre_linear, res=args.res, ln=args.ln,
                       bn=args.bn, jk=args.jk, gnn=args.gnn).to(device)
    return model
        

def parser_add_main_args(parser):
    # meta-learning lambda
    parser.add_argument('--learn_lambda_meta', action='store_true',
                        help='learn lambda online with a meta-network')
    parser.add_argument('--meta_lr', type=float, default=1e-3,
                        help='learning rate for lambda meta-network')
    parser.add_argument('--meta_hidden', type=int, default=32,
                        help='hidden dimension for lambda meta-network')
    parser.add_argument('--lambda_max', type=float, default=1.0,
                        help='maximum lambda predicted by meta-network')
    parser.add_argument('--meta_warmup_epochs', type=int, default=20,
                        help='epochs before starting meta-learning')
    parser.add_argument('--learn_lambda_meta_v2', action='store_true',
                        help='learn lambda jointly with the network parameters on the training loss; no bilevel optimization')
    parser.add_argument('--learn_lambda_meta_v3', action='store_true',
                        help='split train labels into inner/meta sets and run bilevel optimization on the held-out train subset, not the validation set')
    parser.add_argument('--meta_train_holdout_frac', type=float, default=0.10,
                        help='fraction of train labels used as the meta split for --learn_lambda_meta_v3')
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='roman-empire')
    parser.add_argument('--data_dir', type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--rand_split', action='store_true',
                        help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=1000,
                        help='Total number of test')
    
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc'],
                        help='evaluation metric')
    parser.add_argument('--model', type=str, default='MPNN')
    # GNN
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--local_layers', type=int, default=7)
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--pre_ln', action='store_true')
    parser.add_argument('--pre_linear', action='store_true')
    parser.add_argument('--res', action='store_true', help='use residual connections for GNNs')
    parser.add_argument('--ln', action='store_true', help='use normalization for GNNs')
    parser.add_argument('--bn', action='store_true', help='use normalization for GNNs')
    parser.add_argument('--jk', action='store_true', help='use JK for GNNs')
    
    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    # entropy regularization
    parser.add_argument('--lambda_u', type=float, default=0.0,
                        help='entropy weight on unlabeled nodes')
    parser.add_argument('--lambda_t', type=float, default=0.0,
                        help='entropy weight on labeled (train) nodes')
    parser.add_argument('--tie_lambda_t', action='store_true',
                        help='set lambda_t = -lambda_u')
    parser.add_argument('--midpoint_offset', type=float, default=0.0,
                        help='shift (lambda_u, lambda_t) midpoint by this amount '
                             'while preserving spread; applied AFTER --tie_lambda_t')
    parser.add_argument('--warmup_frac', type=float, default=0.0,
                        help='fraction of epochs to linearly ramp entropy reg (0=off)')
    parser.add_argument('--tsallis_q', type=float, default=1.0,
                        help='Tsallis entropy parameter (1=Shannon, 2+=heavier tails)')
    parser.add_argument('--mi-max', type=float, default=0.0,
                        help='MI class-balance weight: subtract mi_max * H(Y_marginal) (0=off, 1=standard MI)')
    parser.add_argument('--lambda_val', type=float, default=None,
                        help='entropy weight on val nodes (default: same as --lambda_u)')
    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=100, help='how often to print')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--model_dir', type=str, default='./model/', help='where to save model')


