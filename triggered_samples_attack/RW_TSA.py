import warnings

import pandas as pd

warnings.filterwarnings("ignore")

import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from triggered_samples_attack.utils import *
from models.quantization import *
from models import quan_resnet
import numpy as np
import config
from models.model_wrap import Attacked_model
import copy
import argparse
from real_world_augmentations import RWAugmentations
import kornia.augmentation as K
import math
from custom_nets.lit_modules import BasicLitModule
from torch.nn import functional as F


parser = argparse.ArgumentParser(description='Triggered Samples Attack')

parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--trigger-size', dest='trigger_size', type=int, default=10)
parser.add_argument('--target', dest='target', type=int, default=0)
parser.add_argument('--dataset_type', dest='dataset_type', type=str, default='cifar10')
parser.add_argument('--results_output_path', type=str, default='results.xlsx')

parser.add_argument('--gpu-id', dest='gpu_id', type=str, default='0')

parser.add_argument('--lams1', dest='lams1', default=[65], nargs='+', type=int)
parser.add_argument('--lam2', dest='lam2', default=1, type=float)
parser.add_argument('--k-bits', '-k_bits', default=[30], nargs='+', type=int)
parser.add_argument('--n-aux', '-n_aux', default=256, type=int)
parser.add_argument('--remove_second_phase', dest='remove_second_phase', action='store_true')

parser.add_argument('--max-search', '-max_search', default=8, type=int)
parser.add_argument('--ext-max-iters', '-ext_max_iters', default=2000, type=int)
parser.add_argument('--inn-max-iters', '-inn_max_iters', default=5, type=int)
parser.add_argument('--initial-rho1', '-initial_rho1', default=0.0001, type=float)
parser.add_argument('--initial-rho2', '-initial_rho2', default=0.0001, type=float)
parser.add_argument('--initial-rho3', '-initial_rho3', default=0.00001, type=float)
parser.add_argument('--max-rho1', '-max_rho1', default=100, type=float)
parser.add_argument('--max-rho2', '-max_rho2', default=100, type=float)
parser.add_argument('--max-rho3', '-max_rho3', default=10, type=float)
parser.add_argument('--rho-fact', '-rho_fact', default=1.01, type=float)
parser.add_argument('--inn-lr-bit', '-inn_lr_bit', default=0.001, type=float)
parser.add_argument('--inn-lr-trigger', '-inn_lr_trigger', default=1, type=float)
parser.add_argument('--stop-threshold', '-stop_threshold', default=1e-4, type=float)
parser.add_argument('--projection-lp', '-projection_lp', default=2, type=int)

parser.add_argument("--silent", action='store_true', help='execute attack silently')

args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
print(f"GPU:{torch.cuda.is_available()}")

print("Prepare data ... ")

if args.dataset_type == 'cifar10':
    dataset_dir = config.cifar_root
    val_set = datasets.CIFAR10(root=dataset_dir, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    class_num = 10
elif args.dataset_type == 'gtsrb':
    dataset_dir = config.gtsrb_root
    gtsrb_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    val_set = datasets.GTSRB(root=dataset_dir, split='test', transform=gtsrb_transform)
    normalize = transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))
    class_num = 43
else:
    raise NotImplementedError

val_loader = torch.utils.data.DataLoader(
    dataset=val_set,
    batch_size=args.batch_size, shuffle=False, pin_memory=True)

input_size = 32
bit_length = 8
model = torch.nn.DataParallel(quan_resnet.resnet20_quan_mid(class_num, bit_length))
if args.dataset_type == 'cifar10':

    checkpoint = torch.load(config.cifar_model_path)
    model.load_state_dict(checkpoint["state_dict"])
else:
    checkpoint_path = config.gtsrb_model_path
    lit_model = BasicLitModule(model, F.cross_entropy)
    checkpoint = torch.load(checkpoint_path)
    lit_model.load_state_dict(checkpoint["state_dict"])
    model.load_state_dict(lit_model.model.state_dict())


if isinstance(model, torch.nn.DataParallel):
    model = model.module

for m in model.modules():
    if isinstance(m, quan_Linear):
        m.__reset_stepsize__()
        m.__reset_weight__()
model.cuda()

load_model = Attacked_model(model, "resnet20_quan_8")
load_model.cuda()
load_model.eval()

model = torch.nn.DataParallel(model)
load_model.model = torch.nn.DataParallel(load_model.model)

criterion = nn.CrossEntropyLoss().cuda()


n_aux = args.n_aux  # the size of auxiliary sample set
lam2 = args.lam2
ext_max_iters = args.ext_max_iters
inn_max_iters = args.inn_max_iters
initial_rho1 = args.initial_rho1
initial_rho2 = args.initial_rho2
initial_rho3 = args.initial_rho3
max_rho1 = args.max_rho1
max_rho2 = args.max_rho2
max_rho3 = args.max_rho3
rho_fact = args.rho_fact
inn_lr_bit = args.inn_lr_bit
inn_lr_trigger = args.inn_lr_trigger
stop_threshold = args.stop_threshold

projection_lp = args.projection_lp

target_class = args.target

l_transforms = [
    transforms.RandomRotation(degrees=(10, 10)),
    transforms.RandomHorizontalFlip(p=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
]
transform = transforms.Compose(l_transforms)
 
if args.dataset_type == 'cifar10':
    np.random.seed(512)
    aux_idx = np.random.choice(len(val_loader.dataset), args.n_aux, replace=False)
    aux_dataset = ImageFolder_cifar10(val_loader.dataset.data[aux_idx],
                                      np.array(val_loader.dataset.targets)[aux_idx],
                                      transform=transform)
else:
    aux_dataset = datasets.GTSRB(root=dataset_dir, split='test', transform=transform)

    # make the dataset smaller with args.n_aux samples:
    aux_dataset = torch.utils.data.Subset(aux_dataset, list(range((args.n_aux))))

aux_loader = torch.utils.data.DataLoader(
    dataset=aux_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=True)

rw_transforms = [
    # fill RW augs
    K.RandomMotionBlur(3, 35., 0.5, p=1., same_on_batch=True),
    K.RandomPerspective(distortion_scale=0.25, p=1, sampling_method='area_preserving', same_on_batch=True, align_corners=True),
    K.RandomBrightness((0.75, 1.25), p=1, same_on_batch=True),
]

non_diff_rw_transforms = [
    # fill RW augs
    K.RandomErasing((.005, .015), (.3, 3.3), p=1, same_on_batch=True),
]

rw_augs = RWAugmentations(rw_transforms, p=0.5)
non_diff_rw_augs = RWAugmentations(non_diff_rw_transforms, p=0.5)


def pnorm(x, p=2):
    batch_size = x.size(0)
    norm = x.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(1. / p)
    norm = torch.max(norm, torch.ones_like(norm) * 1e-6)
    return norm


def loss_func(output, labels, output_trigger, labels_trigger, lam1, lam2, w,
              b_ori, k_bits, y1, y2, y3, z1, z2, z3, k, rho1, rho2, rho3):

    l1 = F.cross_entropy(output_trigger, labels_trigger.type(torch.LongTensor).to('cuda'))
    l2 = F.cross_entropy(output, labels.type(torch.LongTensor).to('cuda'))

    y1, y2, y3, z1, z2, z3 = torch.tensor(y1).float().cuda(), torch.tensor(y2).float().cuda(), torch.tensor(y3).float().cuda(), \
                             torch.tensor(z1).float().cuda(), torch.tensor(z2).float().cuda(), torch.tensor(z3).float().cuda()

    b_ori = torch.tensor(b_ori).float().cuda()
    b = w.view(-1)

    l3 = z1@(b-y1) + z2@(b-y2) + z3*(torch.norm(b - b_ori) ** 2 - k + y3)

    l4 = (rho1/2) * torch.norm(b - y1) ** 2 + (rho2/2) * torch.norm(b - y2) ** 2 \
       + (rho3/2) * (torch.norm(b - b_ori)**2 - k_bits + y3) ** 2

    return lam1 * l1 + lam2 * l2 + l3 + l4, l1.item(), l2.item()


def attack_func_orig(k_bits, lam1, lam2):
    attacked_model = copy.deepcopy(load_model)
    attacked_model_ori = copy.deepcopy(load_model)

    trigger = torch.randn([1, 3, input_size, input_size]).float().cuda()
    trigger_mask = torch.zeros([1, 3, input_size, input_size]).cuda()
    trigger_mask[:, :, input_size-args.trigger_size:input_size, input_size-args.trigger_size:input_size] = 1

    trigger = optimization_loop(attacked_model, None, k_bits, lam1, lam2,
                                trigger, trigger_mask, ext_max_iters_loop=ext_max_iters * 2,
                                optimize_trigger=True)
    n_bit = torch.norm(attacked_model.w_twos.data.view(-1) - attacked_model_ori.w_twos.data.view(-1), p=0).item()

    clean_acc_auged, _, _ = validate(val_loader, nn.Sequential(rw_augs, non_diff_rw_augs, normalize, attacked_model), criterion)

    trigger_acc_auged, _, _ = validate_trigger(val_loader, trigger, trigger_mask, target_class,
                                         nn.Sequential(rw_augs, non_diff_rw_augs, normalize, attacked_model), criterion)

    aux_trigger_acc_auged, _, _ = validate_trigger(aux_loader, trigger, trigger_mask, target_class,
                                             nn.Sequential(rw_augs, non_diff_rw_augs, normalize, attacked_model), criterion)

    clean_acc, _, _ = validate(val_loader, nn.Sequential(normalize, attacked_model), criterion)

    trigger_acc, _, _ = validate_trigger(val_loader, trigger, trigger_mask, target_class,
                                         nn.Sequential(normalize, attacked_model), criterion)

    aux_trigger_acc, _, _ = validate_trigger(aux_loader, trigger, trigger_mask, target_class,
                                             nn.Sequential(normalize, attacked_model), criterion)

    return attacked_model, clean_acc, trigger_acc, n_bit, aux_trigger_acc, clean_acc_auged, trigger_acc_auged, aux_trigger_acc_auged


def attack_func_real_world(k_bits, lam1, lam2, remove_second_phase=False):

    attacked_model = copy.deepcopy(load_model)
    attacked_model_ori = copy.deepcopy(load_model)

    trigger = torch.randn([1, 3, input_size, input_size]).float().cuda()
    trigger_mask = torch.zeros([1, 3, input_size, input_size]).cuda()
    trigger_mask[:, :, input_size-args.trigger_size:input_size, input_size-args.trigger_size:input_size] = 1

    if remove_second_phase:
        trigger = optimization_loop(attacked_model, rw_augs, k_bits, lam1, lam2,
                                    trigger, trigger_mask, ext_max_iters_loop=ext_max_iters,
                                    optimize_trigger=True)
    else:
        trigger = optimization_loop(attacked_model, rw_augs, math.ceil(k_bits / 2), lam1, lam2,
                                    trigger, trigger_mask, ext_max_iters_loop=ext_max_iters // 2,
                                    optimize_trigger=True)

        optimization_loop(attacked_model, nn.Sequential(rw_augs, non_diff_rw_augs), math.floor(k_bits / 2), lam1, lam2,
                          trigger, trigger_mask, ext_max_iters_loop=ext_max_iters // 2,
                          optimize_trigger=False)

    n_bit = torch.norm(attacked_model.w_twos.data.view(-1) - attacked_model_ori.w_twos.data.view(-1), p=0).item()

    clean_acc_auged, _, _ = validate(val_loader,
                               nn.Sequential(rw_augs, non_diff_rw_augs, normalize, attacked_model), criterion)

    trigger_acc_auged, _, _ = validate_trigger(val_loader, trigger, trigger_mask, target_class,
                                         nn.Sequential(rw_augs, non_diff_rw_augs, normalize, attacked_model), criterion)

    aux_trigger_acc_auged, _, _ = validate_trigger(aux_loader, trigger, trigger_mask, target_class,
                                             nn.Sequential(rw_augs, non_diff_rw_augs, normalize, attacked_model), criterion)

    clean_acc, _, _ = validate(val_loader, nn.Sequential(normalize, attacked_model), criterion)

    trigger_acc, _, _ = validate_trigger(val_loader, trigger, trigger_mask, target_class,
                                         nn.Sequential(normalize, attacked_model), criterion)

    aux_trigger_acc, _, _ = validate_trigger(aux_loader, trigger, trigger_mask, target_class,
                                             nn.Sequential(normalize, attacked_model), criterion)

    return attacked_model, clean_acc, trigger_acc, n_bit, aux_trigger_acc, clean_acc_auged, trigger_acc_auged, aux_trigger_acc_auged


def optimization_loop(attacked_model, real_world_augs, k_bits, lam1, lam2,
                      trigger, trigger_mask, ext_max_iters_loop,
                      optimize_trigger=True):
    b_ori = attacked_model.w_twos.data.view(-1).detach().cpu().numpy()
    b_new = b_ori

    z1 = np.zeros_like(b_ori)
    z2 = np.zeros_like(b_ori)
    z3 = 0

    rho1 = initial_rho1
    rho2 = initial_rho2
    rho3 = initial_rho3

    for ext_iter in range(ext_max_iters_loop):

        y1 = project_box(b_new + z1 / rho1)
        y2 = project_shifted_Lp_ball(b_new + z2 / rho2, projection_lp)
        y3 = project_positive(-np.linalg.norm(b_new - b_ori, ord=2) ** 2 + k_bits - z3 / rho3)

        for inn_iter in range(inn_max_iters):

            for i, (input, label) in enumerate(aux_loader):
                input_var = torch.autograd.Variable(input, volatile=True).cuda()
                label_var = torch.autograd.Variable(label, volatile=True).cuda()

                target_trigger_var = torch.zeros_like(label_var) + target_class
                trigger = torch.autograd.Variable(trigger, requires_grad=optimize_trigger)

                input_triggered = input_var * (1 - trigger_mask) + trigger * trigger_mask
                # concatenate batches of input_var and input_triggered:
                # Note that every batch will be augmented with the same augmentations:
                input_var_and_input_triggered = torch.cat((input_var, input_triggered), 0)
                if real_world_augs is not None:
                    input_var_and_input_triggered_aug = real_world_augs(input_var_and_input_triggered)
                else:
                    input_var_and_input_triggered_aug = input_var_and_input_triggered
                input_var_aug = input_var_and_input_triggered_aug[:input_var.shape[0]]
                input_triggered_aug = input_var_and_input_triggered_aug[input_var.shape[0]:]

                output = attacked_model(normalize(input_var_aug))
                output_triggered = attacked_model(normalize(input_triggered_aug))

                loss, loss1, loss2 = loss_func(output, label_var, output_triggered, target_trigger_var,
                                               lam1, lam2, attacked_model.w_twos,
                                               b_ori, k_bits, y1, y2, y3, z1, z2, z3, k_bits, rho1, rho2, rho3)

                loss.backward(retain_graph=True)

                attacked_model.w_twos.data = attacked_model.w_twos.data - \
                                             inn_lr_bit * attacked_model.w_twos.grad.data
                if optimize_trigger:
                    if ext_iter < 1000:
                        trigger.data = trigger.data - inn_lr_trigger * trigger.grad.data
                    else:
                        trigger.data = trigger.data - inn_lr_trigger * 0.1 * trigger.grad.data

                for name, param in attacked_model.named_parameters():
                    if param.grad is not None:
                        param.grad.detach_()
                        param.grad.zero_()
                if optimize_trigger:
                    trigger.grad.zero_()

                    trigger = torch.clamp(trigger, min=0.0, max=1.0)

        b_new = attacked_model.w_twos.data.view(-1).detach().cpu().numpy()

        z1 = z1 + rho1 * (b_new - y1)
        z2 = z2 + rho2 * (b_new - y2)
        z3 = z3 + rho3 * (np.linalg.norm(b_new - b_ori, ord=2) ** 2 - k_bits + y3)

        rho1 = min(rho_fact * rho1, max_rho1)
        rho2 = min(rho_fact * rho2, max_rho2)
        rho3 = min(rho_fact * rho3, max_rho3)

        temp1 = (np.linalg.norm(b_new - y1)) / max(np.linalg.norm(b_new), 2.2204e-16)
        temp2 = (np.linalg.norm(b_new - y2)) / max(np.linalg.norm(b_new), 2.2204e-16)
        if ext_iter % 400 == 0 and not args.silent:
            print('iter: %d, stop_threshold: %.6f loss: %.4f' % (
                ext_iter, max(temp1, temp2), loss.item()))

        if max(temp1, temp2) <= stop_threshold and ext_iter > 100:
            break

    attacked_model.w_twos.data[attacked_model.w_twos.data > 0.5] = 1.0
    attacked_model.w_twos.data[attacked_model.w_twos.data < 0.5] = 0.0

    return trigger


def main():
    orig_acc, _, _ = validate(val_loader, nn.Sequential(normalize, load_model), criterion)
    orig_acc_auged, _, _ = validate(val_loader, nn.Sequential(rw_augs, non_diff_rw_augs, normalize, load_model), criterion)
    print("Original Acc: {0:.4f}, Original Augmented Acc: {1:.4f}".format(orig_acc, orig_acc_auged))
    all_k_bits = args.k_bits if isinstance(args.k_bits, list) else [args.k_bits]
    all_lam1 = args.lams1 if isinstance(args.lams1, list) else [args.lams1]
    # Create an empty DataFrame to store the results
    results_df_cols = [
        'k_bits',
        'lam1',
        'clean_acc_orig',
        'trigger_acc_orig',
        'n_bit_orig',
        'aux_trigger_acc_orig',
        'clean_acc_auged_orig',
        'trigger_acc_auged_orig',
        'aux_trigger_acc_auged_orig',
        'clean_acc_rw',
        'trigger_acc_rw',
        'n_bit_rw',
        'aux_trigger_acc_rw',
        'clean_acc_auged_rw',
        'trigger_acc_auged_rw',
        'aux_trigger_acc_auged_rw'
    ]
    results_df_rows = []

    # Loop through all combinations of k_bits and lam1
    for k_bits in all_k_bits:
        for lam1 in all_lam1:
            # Call your functions to get the results
            print("Original Attack Start, k =", k_bits)
            attacked_model_orig, clean_acc_orig, trigger_acc_orig, n_bit_orig, aux_trigger_acc_orig, \
                clean_acc_auged_orig, trigger_acc_auged_orig, aux_trigger_acc_auged_orig = attack_func_orig(k_bits,
                                                                                                            lam1, lam2)
            print("non augmented metrics:")
            print(f"aux_trigger_acc: {aux_trigger_acc_orig:.4f}")
            print("target:{0} clean_acc:{1:.4f} asr:{2:.4f} bit_flips:{3}".format(
                args.target, clean_acc_orig, trigger_acc_orig, n_bit_orig))
            print("augmented metrics:")
            print(f"aux_trigger_acc: {aux_trigger_acc_auged_orig:.4f}")
            print("target:{0} clean_acc:{1:.4f} asr:{2:.4f} bit_flips:{3}".format(
                args.target, clean_acc_auged_orig, trigger_acc_auged_orig, n_bit_orig))

            print("Real World Attack Start, k =", k_bits)
            attacked_model_rw, clean_acc_rw, trigger_acc_rw, n_bit_rw, aux_trigger_acc_rw, \
                clean_acc_auged_rw, trigger_acc_auged_rw, aux_trigger_acc_auged_rw = attack_func_real_world(k_bits,
                                                                                                            lam1, lam2,
                                                                                                            args.remove_second_phase)
            print("non augmented metrics:")
            print(f"aux_trigger_acc: {aux_trigger_acc_rw:.4f}")
            print("target:{0} clean_acc:{1:.4f} asr:{2:.4f} bit_flips:{3}".format(
                args.target, clean_acc_rw, trigger_acc_rw, n_bit_rw))
            print("augmented metrics:")
            print(f"aux_trigger_acc: {aux_trigger_acc_auged_rw:.4f}")
            print("target:{0} clean_acc:{1:.4f} asr:{2:.4f} bit_flips:{3}".format(
                args.target, clean_acc_auged_rw, trigger_acc_auged_rw, n_bit_rw))
            # Create a dictionary with the results
            results_df_rows.append({
                'k_bits': k_bits,
                'lam1': lam1,
                'PA_ACC_TSA': clean_acc_orig,
                'ASR_TSA': trigger_acc_orig,
                'n_bit_TSA': n_bit_orig,
                'aux_ASR_TSA': aux_trigger_acc_orig,
                'PA_ACC_auged_TSA': clean_acc_auged_orig,
                'ASR_auged_TSA': trigger_acc_auged_orig,
                'aux_ASR_auged_TSA': aux_trigger_acc_auged_orig,
                'PA_ACC_TSA_RW': clean_acc_rw,
                'ASR_TSA_RW': trigger_acc_rw,
                'n_bit_TSA_RW': n_bit_rw,
                'aux_ASR_TSA_RW': aux_trigger_acc_rw,
                'PA_ACC_auged_TSA_RW': clean_acc_auged_rw,
                'ASR_auged_TSA_RW': trigger_acc_auged_rw,
                'aux_ASR_auged_TSA_RW': aux_trigger_acc_auged_rw
            })

            # Append the result dictionary to the DataFrame
    results_df = pd.DataFrame(data=results_df_rows, columns=results_df_cols)

    # Save the DataFrame to an Excel file
    results_df.to_excel(args.results_output_path, index=False, engine='openpyxl')


if __name__ == '__main__':
    main()