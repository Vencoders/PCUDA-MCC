# -*- coding: UTF-8 -*-
import numpy as np
import random
import json
import math
import torch
import operator
from collections import Counter
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from utils.pc_utils import random_rotate_one_axis
import sklearn.metrics as metrics
import argparse
import copy
import utils.log
from data.dataloader_Norm import ScanNet, ModelNet, ShapeNet, label_to_idx, NUM_POINTS
from Models_Norm import PointNet, DGCNN

import moco.builder

# from fullmatch_utils import cal_topK, nl_em_loss


# from utils.pc_utils_Norm import Mixup1

import nt_xent
from utils import pc_utils_Norm, loss, log




NWORKERS=4
MAX_LOSS = 9 * (10**9)
threshold = 0.8
spl_weight = 1
cls_weight = 1



def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# ==================
# Argparse
# ==================
parser = argparse.ArgumentParser(description='DA on Point Clouds')
parser.add_argument('--exp_name', type=str, default='GAST_SPST',  help='Name of the experiment')
parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
parser.add_argument('--dataroot', type=str, default='./data', metavar='N', help='data path')
parser.add_argument('--model_file', type=str, default='model_75.68.ptdgcnn', help='pretrained model file')
parser.add_argument('--src_dataset', type=str, default='scannet', choices=['modelnet', 'shapenet', 'scannet'])
parser.add_argument('--trgt_dataset', type=str, default='shapenet', choices=['modelnet', 'shapenet', 'scannet'])
parser.add_argument('--epochs', type=int, default=20, help='number of episode to train')
parser.add_argument('--model', type=str, default='dgcnn', choices=['pointnet', 'dgcnn'], help='Model to use')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='3',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--DefRec_dist', type=str, default='volume_based_voxels', metavar='N',
                    choices=['volume_based_voxels', 'volume_based_radius'],
                    help='distortion of points')
parser.add_argument('--num_regions', type=int, default=3, help='number of regions to split shape by')
parser.add_argument('--DefRec_on_src', type=str2bool, default=False, help='Using DefRec in source')
parser.add_argument('--DefRec_on_trgt', type=str2bool, default=False, help='Using DefRec in target')
parser.add_argument('--DefCls_on_src', type=str2bool, default=False, help='Using DefCls in source')
parser.add_argument('--DefCls_on_trgt', type=str2bool, default=False, help='Using DefCls in target')
parser.add_argument('--PosReg_on_src', type=str2bool, default=False, help='Using PosReg in source')
parser.add_argument('--PosReg_on_trgt', type=str2bool, default=False, help='Using PosReg in target')
parser.add_argument('--apply_PCM', type=str2bool, default=False, help='Using mixup in source')
parser.add_argument('--apply_GRL', type=str2bool, default=False, help='Using gradient reverse layer')
parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size', help='Size of train batch per domain')
parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size', help='Size of test batch per domain')
parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
parser.add_argument('--cls_weight', type=float, default=0.5, help='weight of the classification loss')
parser.add_argument('--grl_weight', type=float, default=0.5, help='weight of the GRL loss')
parser.add_argument('--DefRec_weight', type=float, default=0.5, help='weight of the DefRec loss')
parser.add_argument('--DefCls_weight', type=float, default=0.5, help='weight of the DefCls loss')
parser.add_argument('--PosReg_weight', type=float, default=0.5, help='weight of the PosReg loss')
parser.add_argument('--output_pts', type=int, default=512, help='number of decoder points')
parser.add_argument('--mixup_params', type=float, default=1.0, help='a,b in beta distribution')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--temperature', default=0.5, type=float, help='temperature')
parser.add_argument('--zeta', default=0.1, type=float, help='variance')

parser.add_argument('--tool', default="orig", type=str, help="orig/RC/LBE/IP/MIB/BYOL")
parser.add_argument('--p_cutoff', type=float, default=0.95)

parser.add_argument(
    "--moco-k",
    default=65536,
    type=int,
    help="queue size; number of negative keys (default: 65536)",
)
parser.add_argument(
    "--moco-m",
    default=0.999,
    type=float,
    help="moco momentum of updating key encoder (default: 0.999)",
)
parser.add_argument(
    "--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)"
)

args = parser.parse_args()

# ==================
# init
# ==================
io = utils.log.IOStream(args)
io.cprint(str(args))

random.seed(1)
# np.random.seed(1)  # to get the same point choice in ModelNet and ScanNet leave it fixed
torch.manual_seed(args.seed)
args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
if args.cuda:
    io.cprint('Using GPUs ' + str(args.gpus) + ',' + ' from ' +
              str(torch.cuda.device_count()) + ' devices available')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    io.cprint('Using CPU')

# ==================
# Init Model
# ==================
# if args.model == 'pointnet':
#     model = PointNet(args)
#     model.load_state_dict(torch.load('./experiments/GAST/model.ptpointnet'))
# elif args.model == 'dgcnn':
#     model = DGCNN(args)
#     model.load_state_dict(torch.load('./experiments/GAST/' + args.model_file))
# else:
#     raise Exception("Not implemented")


model = moco.builder.MoCo_Model(args, queue_size=args.moco_k,
                      momentum=args.moco_m, temperature=args.moco_t)

model_orig = model.to(device)

model.load_state_dict(torch.load('./experiments/GAST/' + args.model_file ,map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())))

model = model.to(device)

# Handle multi-gpu
if (device.type == 'cuda') and len(args.gpus) > 1:
    model = nn.DataParallel(model, args.gpus)
best_model = copy.deepcopy(model)

src_val_acc_list = []
src_val_loss_list = []
trgt_val_acc_list = []
trgt_val_loss_list = []


# ==================
# loss function
# ==================
opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd) if args.optimizer == "SGD" \
    else optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = CosineAnnealingLR(opt, args.epochs)
criterion = nn.CrossEntropyLoss()  # return the mean of CE over the batch

similarity_f = nn.CosineSimilarity(dim=2)

# ==================
# Read Data
# ==================
def split_set(dataset, domain, set_type="source"):
    """
    Input:
        dataset
        domain - modelnet/shapenet/scannet
        type_set - source/target
    output:
        train_sampler, valid_sampler
    """
    train_indices = dataset.train_ind
    val_indices = dataset.val_ind
    unique, counts = np.unique(dataset.label[train_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " train part: " + str(dict(zip(unique, counts))))
    unique, counts = np.unique(dataset.label[val_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " validation part: " + str(dict(zip(unique, counts))))
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler

trgt_dataset = args.trgt_dataset
src_dataset = args.src_dataset

data_func = {'modelnet': ModelNet, 'scannet': ScanNet, 'shapenet': ShapeNet}

src_trainset = data_func[src_dataset](io, args.dataroot, 'train')
trgt_trainset = data_func[trgt_dataset](io, args.dataroot, 'train')
trgt_testset = data_func[trgt_dataset](io, args.dataroot, 'test')

src_train_sampler, src_valid_sampler = split_set(src_trainset, src_dataset, "source")
trgt_train_sampler, trgt_valid_sampler = split_set(trgt_trainset, trgt_dataset, "target")

# dataloaders for finetue and test
src_train_loader = DataLoader(src_trainset, num_workers=NWORKERS, batch_size=args.batch_size,
                              sampler=src_train_sampler, drop_last=True)
src_val_loader = DataLoader(src_trainset, num_workers=NWORKERS, batch_size=args.test_batch_size,
                            sampler=src_valid_sampler)
trgt_train_loader = DataLoader(trgt_trainset, num_workers=NWORKERS, batch_size=args.batch_size,
                               sampler=trgt_train_sampler, drop_last=True)
trgt_val_loader = DataLoader(trgt_trainset, num_workers=NWORKERS, batch_size=args.test_batch_size,
                             sampler=trgt_valid_sampler)
trgt_test_loader = DataLoader(trgt_testset, num_workers=NWORKERS, batch_size=args.test_batch_size)


def entropy(*c):
    result = -1
    if len(c) > 0:
        result = 0
    for x in c:
        result += (-x) * math.log(x, 2)
    return result

# ==================
# select_target_data
# ==================

def trim_mean(data, proportion_to_cut):
    """
    去除最差和最好的一定比例的数据，然后计算剩余数据的平均值
    :param data: 输入数据
    :param proportion_to_cut: 要去除的数据比例（0-0.5）
    :return: 去除后的平均值
    """
    # 确定要去除的数据点数量
    n = len(data)
    k = int(proportion_to_cut * n)

    # 排序数据
    sorted_data = np.sort(data)

    # 去除最差和最好的部分
    trimmed_data = sorted_data[k:n - k]

    # 计算去除后的平均值
    trimmed_mean = np.mean(trimmed_data)
    return trimmed_mean

def obtain_threshold_by_conf(trgt_train_loader, model=None):

    threshold_list = []
    tl = [[] for _ in range(10)]

    sfm = nn.Softmax(dim=1)

    with torch.no_grad():
        model.eval()
        for data in trgt_train_loader:
            data = data[0].to(device)
            data = data.permute(0, 2, 1)

            logits,_ = model(data, data, data,device, activate_N=True)

            cls_conf = sfm(logits['cls'])
            mask = torch.max(cls_conf, 1)  # 2 * b

            for i, c in enumerate(mask[1]):
                tl[c].append(mask[0][i].cpu().numpy())



        threshold_list = [np.mean(tl[i]) for i in range(10)]

        for i in range(10):
            print(np.std(tl[i]))

        # threshold_list = [np.mean(tl[i]) + 0.01 * np.std(tl[i]) for i in range(10)]

        print("==",threshold_list)

        # 计算平均值
        mean_val = np.mean(threshold_list)
        # 调整因子 (0到1之间，值越接近1靠拢程度越高)
        alpha = 0.4
        # 调整阈值
        adjusted_thresholds = [(1 - alpha) * threshold + alpha * mean_val for threshold in threshold_list]


        # threshold_list = [trim_mean(tl[i], 0.05) for i in range(10)]


        # threshold_list = [np.median(tl[i]) for i in range(10)]

        # threshold_list = [np.percentile(tl[i], 30) for i in range(10)]

        # for i in range(10):
        #     threshold_list[i] = 2 * threshold_list[i] / (threshold_list[i] +1)

        # for i in range(10):
        #     threshold_list[i] = threshold_list[i] ** 1.5
            # threshold_list[i] = -1 * threshold_list[i] / (threshold_list[i] - 2)


    return adjusted_thresholds

torch.set_printoptions(threshold=10000)

def weightPro(probs, feat):

    # sfm = nn.Softmax(dim=1)
    # probs = sfm(logits['cls'])
    sim = similarity_f(feat.unsqueeze(1), feat.unsqueeze(0))
    sim = (1 + sim) / 2
    # sim = torch.round(sim * 1000) / 1000

    _, idxs = sim.sort(descending=True)

    # 邻居样本的个数
    k = 1
    idxs = idxs[:, 0: k + 1]

    # 使用相似度作为权重对最近的邻居样本的分布预测进行加权平均
    weighted_probs = torch.zeros(probs.size(0), probs.size(1)).to(probs.device)  # 初始化加权概率张量

    for i in range(probs.size(0)):
        nearest_idxs = idxs[i]  # 获取第i个样本的最近邻居索引
        nearest_probs = probs[nearest_idxs]  # 获取最近邻居的预测分布
        # print(i,nearest_probs)
        nearest_sims = sim[i, nearest_idxs]  # 获取相应的相似度作为权重

        # p是期望中心样本大概占的比例 w是给中心样本的初始权重
        p = 0.8

        w = (p * k) / (1 - p)

        # 对自身相似度进行加权
        nearest_sims[0] *= 5
        normalized_sims = nearest_sims / nearest_sims.sum()  # 将相似度归一化
        weighted_probs[i] = (nearest_probs * normalized_sims.unsqueeze(1)).sum(dim=0)

    return weighted_probs


def select_target_by_conf(trgt_train_loader, thre, model=None):
    pc_list = []
    label_list = []
    sfm = nn.Softmax(dim=1)

    count_a = 0
    count_s = 0
    ac = 0.0

    with torch.no_grad():
        model.eval()
        for data in trgt_train_loader:
            data, label, data_w = data[0].to(device), data[1].to(device), data[4].to(device)

            data = data.permute(0, 2, 1)
            # data_w = data_w.permute(0, 2, 1)

            # logits,_ = model(data, activate_DefRec=False)
            logits, feat = model(data, data, data,device, activate_N=True)

            # weighted_probs = weightPro(logits,feat)
            # mask = torch.max(weighted_probs, 1)  # 2 * b

            cls_conf = sfm(logits['cls'])
            # 邻居一致性
            cls_conf = weightPro(cls_conf, feat)
            mask = torch.max(cls_conf, 1)  # 2 * b

            # cls_conf_w = sfm(logits_w['cls'])
            # mask_w = torch.max(cls_conf_w, 1)  # 2 * b


            index = 0
            for i in mask[0]:
                if i > thre[mask[1][index]]:
                # if i > max(thre[mask[1][index]], 0.75):
                # if i > max(min(thre[mask[1][index]],0.85), 0.6):
                # if i > threshold:
                    pc_list.append(data[index].cpu().numpy())
                    label_list.append(mask[1][index].cpu().numpy())
                    if label[index] == mask[1][index]:
                        count_s += 1
                    count_a += 1
                index += 1

        print(count_s,count_a)
        ac = count_s/count_a
        print("-----",ac)


    return pc_list, label_list




class DataLoad(Dataset):
    def __init__(self, io, data, partition='train'):
        self.partition = partition
        self.pc, self.label = data
        self.num_examples = len(self.pc)

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int32)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int32)
            np.random.shuffle(self.val_ind)

        io.cprint("number of " + partition + " examples in trgt_dataset : " + str(len(self.pc)))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in trgt_dataset " + partition + " set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        pointcloud = np.copy(self.pc[item])
        pointcloud = random_rotate_one_axis(pointcloud.transpose(1, 0), "z")

        pc = pointcloud.copy()

        label = np.copy(self.label[item])
        # print("**",pointcloud.shape)

        # point_m1, point_m2 = Mixup1(pc)

        return (pointcloud, label)

        # return (pointcloud, label, point_m1, point_m2)

    def __len__(self):
        return len(self.pc)


ep = args.epochs


def cross_entropy_loss(pred_probs, labels):
    # 获取真实标签的预测概率
    selected_probs = pred_probs[torch.arange(labels.size(0)), labels]
    # 计算交叉熵损失
    return -torch.mean(torch.log(selected_probs))

def self_train(trgt_new_train_loader, src_train_loader, src_val_loader, trgt_val_loader, model=None):
    count = 0.0
    sfm = nn.Softmax(dim=1)
    src_print_losses = {'cls': 0.0}
    trgt_print_losses = {'cls': 0.0}
    trgt_best_val_acc = 0
    global spl_weight
    global cls_weight
    for epoch in range(ep):
        model.train()
        for data1, data2 in zip(trgt_new_train_loader, src_train_loader):
            opt.zero_grad()
            batch_size = data1[1].size()[0]
            # t_data, t_labels, t1_data, t2_data  = data1[0].to(device), data1[1].to(device),data1[2].to(device), data1[3].to(device)
            t_data, t_labels = data1[0].to(device), data1[1].to(device)


            t_data = t_data.permute(0, 2, 1)
            # t1_data = t1_data.permute(0, 2, 1)
            # t2_data = t2_data.permute(0, 2, 1)

            t_logits, feat = model(t_data, t_data, t_data,device, activate_N=True)
            t_logits_sfm = sfm(t_logits['cls'])

            # 确定预测类别
            predicted_classes = torch.argmax(t_logits_sfm, dim=1)
            # 生成 one-hot 编码
            num_classes = t_logits_sfm.size(1)
            one_hot = torch.zeros(t_logits_sfm.size(0), num_classes).to(t_logits_sfm.device)
            one_hot.scatter_(1, predicted_classes.unsqueeze(1), 1)
            # 按比例相加
            alpha = 0.2
            t_logits_refine = alpha * one_hot + (1 - alpha) * t_logits_sfm

            #细化
            loss_t = spl_weight * cross_entropy_loss(t_logits_refine, t_labels)

            trgt_print_losses['cls'] += loss_t.item() * batch_size
            loss_t.backward()

            count += batch_size
            opt.step()
        spl_weight -= 5e-3  # 0.005
        cls_weight -= 5e-3  # 0.005
        scheduler.step()

        # src_print_losses = {k: v * 1.0 / count for (k, v) in src_print_losses.items()}
        # io.print_progress("Source", "Trn", epoch, src_print_losses)
        trgt_print_losses = {k: v * 1.0 / count for (k, v) in trgt_print_losses.items()}
        io.print_progress("Target_new", "Trn", epoch, trgt_print_losses)
        # ===================
        # Validation
        # ===================
        # src_val_acc, src_val_loss, src_conf_mat = test(src_val_loader, model, "Source", "Val", epoch)
        trgt_val_acc, trgt_val_loss, trgt_conf_mat = test(trgt_val_loader, model, "Target", "Val", epoch)
        # src_val_acc_list.append(src_val_acc)
        # src_val_loss_list.append(src_val_loss)
        trgt_val_acc_list.append(trgt_val_acc)
        trgt_val_loss_list.append(trgt_val_loss)



        if trgt_val_acc > trgt_best_val_acc:
            trgt_best_val_acc = trgt_val_acc
            best_model = io.save_model(model)
            print("-----save best model-----")

        with open('finetune_convergence.json', 'w') as f:
            json.dump((src_val_acc_list, src_val_loss_list, trgt_val_acc_list, trgt_val_loss_list), f)

# ==================
# Validation/test
# ==================
def test(test_loader, model=None, set_type="Target", partition="Val", epoch=0):


    # Run on cpu or gpu
    count = 0.0
    print_losses = {'cls': 0.0}
    batch_idx = 0

    with torch.no_grad():
        model.eval()
        test_pred = []
        test_true = []

        for data, labels, _ in test_loader:

            data, labels = data.to(device), labels.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            # logits,_ = model(data, activate_DefRec=False)

            logits, _ = model(data, data, data,device, activate_N=True)    #凑数

            loss = criterion(logits["cls"], labels)
            print_losses['cls'] += loss.item() * batch_size

            # evaluation metrics
            preds = logits["cls"].max(dim=1)[1]
            test_true.append(labels.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            count += batch_size
            batch_idx += 1

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    print_losses = {k: v * 1.0 / count for (k, v) in print_losses.items()}
    test_acc = io.print_progress(set_type, partition, epoch, print_losses, test_true, test_pred)
    conf_mat = metrics.confusion_matrix(test_true, test_pred, labels=list(label_to_idx.values())).astype(int)

    return test_acc, print_losses['cls'], conf_mat

trgt_test_acc, trgt_test_loss, trgt_conf_mat = test(trgt_test_loader, model, "Target", "Test", 0)
io.cprint("target test accuracy: %.4f, target test loss: %.4f" % (trgt_test_acc, trgt_test_loss))
io.cprint("Test confusion matrix:")
io.cprint('\n' + str(trgt_conf_mat))

# model = copy.deepcopy(best_model)
# trgt_select_data = select_target_by_conf(trgt_train_loader, model)
# trgt_new_data = DataLoad(io, trgt_select_data)
# trgt_new_train_loader = DataLoader(trgt_new_data, num_workers=NWORKERS, batch_size=args.batch_size, drop_last=True)

# model = DGCNN(args)
# model = model.to(device)
# if (device.type == 'cuda') and len(args.gpus) > 1:
#     model = nn.DataParallel(model, args.gpus)
# best_model = copy.deepcopy(model)

# if trgt_test_acc > 0.9:
#     threshold = 0.95
trgt_new_best_val_acc = 0
trgt_new_best_test_acc = 0



for i in range(10):
    model = copy.deepcopy(best_model)
    thre = obtain_threshold_by_conf(trgt_train_loader, model)
    print(thre)
    trgt_select_data = select_target_by_conf(trgt_train_loader, thre, model)
    trgt_new_data = DataLoad(io, trgt_select_data)
    trgt_new_train_loader = DataLoader(trgt_new_data, num_workers=NWORKERS, batch_size=args.batch_size, drop_last=True)


    self_train(trgt_new_train_loader, src_train_loader, src_val_loader, trgt_val_loader, model_orig)
    trgt_new_val_acc, _, _ = test(src_val_loader, model_orig, "Source", "Val", 0)
    # print("trgt_new_val_acc:",trgt_new_val_acc)
    trgt_new_test_acc, _, _ = test(trgt_test_loader, model_orig, "Target", "Test", 0)
    # print("trgt_new_test_acc:",trgt_new_test_acc)

    # if trgt_new_val_acc > trgt_new_best_val_acc:
    #     trgt_new_best_val_acc = trgt_new_val_acc
    best_model = io.save_model_e(model_orig, i)

    print("-----save model-----")

    threshold += 5e-3

trgt_test_acc, trgt_test_loss, trgt_conf_mat = test(trgt_test_loader, best_model, "Target", "Test", 0)
io.cprint("target test accuracy: %.4f, target test loss: %.4f" % (trgt_test_acc, trgt_test_loss))
io.cprint("Test confusion matrix:")
io.cprint('\n' + str(trgt_conf_mat))