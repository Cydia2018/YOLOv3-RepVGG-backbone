import torch
from terminaltables import AsciiTable
from copy import deepcopy
import numpy as np
import torch.nn.functional as F
from scipy.spatial import distance


def get_sr_flag(epoch, sr):
    # return epoch >= 5 and sr
    return sr

def parse_module_defs3(module_defs):

    CBL_idx = []
    Conv_idx = []
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)

    ignore_idx = set()

    ignore_idx.add(18)
    

    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx
    
def parse_module_defs2(module_defs):

    CBL_idx = []
    Conv_idx = []
    shortcut_idx=dict()
    shortcut_all=set()
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)

    ignore_idx = set()
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'shortcut':
            identity_idx = (i + int(module_def['from']))
            if module_defs[identity_idx]['type'] == 'convolutional':
                
                #ignore_idx.add(identity_idx)
                shortcut_idx[i-1]=identity_idx
                shortcut_all.add(identity_idx)
            elif module_defs[identity_idx]['type'] == 'shortcut':
                
                #ignore_idx.add(identity_idx - 1)
                shortcut_idx[i-1]=identity_idx-1
                shortcut_all.add(identity_idx-1)
            shortcut_all.add(i-1)
    #上采样层前的卷积层不裁剪
    ignore_idx.add(84)
    ignore_idx.add(96)

    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx,shortcut_idx,shortcut_all

def parse_module_defs_rep(module_defs):

    CBL_idx = []
    Conv_idx = []
    rep_idx = []
    shortcut_idx=dict()
    shortcut_all=set()
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'RepvggBlock':
            CBL_idx.append(i*2)
            rep_idx.append(i*2)

    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i+28)
            else:
                Conv_idx.append(i+28)

    ignore_idx = set()
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'shortcut':
            identity_idx = (i + int(module_def['from']))
            if module_defs[identity_idx]['type'] == 'convolutional':
                
                #ignore_idx.add(identity_idx)
                shortcut_idx[i-1]=identity_idx
                shortcut_all.add(identity_idx)
            elif module_defs[identity_idx]['type'] == 'shortcut':
                
                #ignore_idx.add(identity_idx - 1)
                shortcut_idx[i-1]=identity_idx-1
                shortcut_all.add(identity_idx-1)
            shortcut_all.add(i-1)
    #上采样层前的卷积层不裁剪
    ignore_idx.add(38+28-1)
    ignore_idx.add(50+28-1)

    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx, rep_idx, shortcut_idx,shortcut_all


def parse_module_defs(module_defs):

    CBL_idx = []
    Conv_idx = []
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)
    ignore_idx = set()
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'shortcut':
            ignore_idx.add(i-1)
            identity_idx = (i + int(module_def['from']))
            if module_defs[identity_idx]['type'] == 'convolutional':
                ignore_idx.add(identity_idx)
            elif module_defs[identity_idx]['type'] == 'shortcut':
                ignore_idx.add(identity_idx - 1)
    #上采样层前的卷积层不裁剪
    ignore_idx.add(84)
    ignore_idx.add(96)

    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx


def gather_bn_weights(module_list, prune_idx):

    size_list = [module_list[idx][1].weight.data.shape[0] for idx in prune_idx]

    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in zip(prune_idx, size_list):
        bn_weights[index:(index + size)] = module_list[idx][1].weight.data.abs().clone()
        index += size

    return bn_weights


def write_cfg(cfg_file, module_defs):

    with open(cfg_file, 'w') as f:
        for module_def in module_defs:
            f.write(f"[{module_def['type']}]\n")
            for key, value in module_def.items():
                if key != 'type':
                    f.write(f"{key}={value}\n")
            f.write("\n")
    return cfg_file


class BNOptimizer():

    @staticmethod
    def updateBN(sr_flag, module_list, s, prune_idx):
        if sr_flag:
            for idx in prune_idx:
                # Squential(Conv, BN, Lrelu)
                bn_module = module_list[idx][1]
                bn_module.weight.grad.data.add_(s * torch.sign(bn_module.weight.data))  # L1


def obtain_quantiles(bn_weights, num_quantile=5):

    sorted_bn_weights, i = torch.sort(bn_weights)
    total = sorted_bn_weights.shape[0]
    quantiles = sorted_bn_weights.tolist()[-1::-total//num_quantile][::-1]
    print("\nBN weights quantile:")
    quantile_table = [
        [f'{i}/{num_quantile}' for i in range(1, num_quantile+1)],
        ["%.3f" % quantile for quantile in quantiles]
    ]
    print(AsciiTable(quantile_table).table)

    return quantiles


def get_input_mask(module_defs, idx, CBLidx2mask):

    if idx == 0:
        return np.ones(3)

    if idx == 56:
        return CBLidx2mask[idx - 2]

    if module_defs[idx-28-1]['type'] == 'convolutional':
        return CBLidx2mask[idx - 1]
    elif module_defs[idx-28-1]['type'] == 'shortcut':
        return CBLidx2mask[idx - 2]
    elif module_defs[idx-28-1]['type'] == 'route':
        # print('idx:')
        # print(idx)
        route_in_idxs = []
        for layer_i in module_defs[idx-28-1]['layers'].split(","):
            if int(layer_i) < 0:
                route_in_idxs.append(idx-1 + int(layer_i))
            else:
                route_in_idxs.append(int(layer_i)*2)
        # print('route_in_idxs:')
        # print(route_in_idxs)
        if len(route_in_idxs) == 1:
            return CBLidx2mask[route_in_idxs[0]]
        elif len(route_in_idxs) == 2:
            # return np.concatenate([CBLidx2mask[in_idx-1] for in_idx in route_in_idxs])
            return np.concatenate([CBLidx2mask[route_in_idxs[0]-1],CBLidx2mask[route_in_idxs[1]]])
        else:
            print("Something wrong with route module!")
            raise Exception

def get_rep_input_mask(module_defs, idx, CBLidx2mask):

    if idx == 0:
        return np.ones(3)

    if module_defs[int(idx/2) - 1]['type'] == 'RepvggBlock':
        return CBLidx2mask[idx - 2]

def init_weights_from_loose_model(compact_model, loose_model, CBL_idx, Conv_idx, CBLidx2mask):

    for idx in CBL_idx:
        compact_CBL = compact_model.module_list[idx]
        loose_CBL = loose_model.module_list[idx]
        out_channel_idx = np.argwhere(CBLidx2mask[idx])[:, 0].tolist()

        compact_bn, loose_bn         = compact_CBL[1], loose_CBL[1]
        compact_bn.weight.data       = loose_bn.weight.data[out_channel_idx].clone()
        compact_bn.bias.data         = loose_bn.bias.data[out_channel_idx].clone()
        compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
        compact_bn.running_var.data  = loose_bn.running_var.data[out_channel_idx].clone()

        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv, loose_conv = compact_CBL[0], loose_CBL[0]
        tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()

    for idx in Conv_idx:
        compact_conv = compact_model.module_list[idx][0]
        loose_conv = loose_model.module_list[idx][0]

        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv.weight.data = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.bias.data   = loose_conv.bias.data.clone()

def init_weights_from_loose_model_rep(compact_model, loose_model, CBL_idx, Conv_idx, rep_idx, CBLidx2mask):

    # print(compact_model.module_list)
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~')
    # print(loose_model.module_list)

    for idx in CBL_idx:
        if idx in rep_idx:
            compact_CBL = compact_model.module_list[idx]
            loose_CBL = loose_model.module_list[idx]
            # print(compact_CBL)
            # print(loose_CBL)
            out_channel_idx = np.argwhere(CBLidx2mask[idx])[:, 0].tolist()

            input_mask = get_rep_input_mask(loose_model.module_defs, idx, CBLidx2mask)
            # print(input_mask)
            # try:
            #     in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
            # except:
            #     print(idx)
            #     print(input_mask)
            in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
            # if idx==0:
            #     print(in_channel_idx)
            #     print('------------')
            #     print(out_channel_idx)
            compact_conv, loose_conv = compact_CBL[0], loose_CBL[0]
            tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
            compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()
            # iden = compact_conv.weight.data==loose_conv.weight.data
            # print(iden.sum())
        else:
            compact_CBL = compact_model.module_list[idx]
            loose_CBL = loose_model.module_list[idx]
            out_channel_idx = np.argwhere(CBLidx2mask[idx])[:, 0].tolist()

            compact_bn, loose_bn         = compact_CBL[1], loose_CBL[1]
            compact_bn.weight.data       = loose_bn.weight.data[out_channel_idx].clone()
            compact_bn.bias.data         = loose_bn.bias.data[out_channel_idx].clone()
            compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
            compact_bn.running_var.data  = loose_bn.running_var.data[out_channel_idx].clone()

            input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
            in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
            compact_conv, loose_conv = compact_CBL[0], loose_CBL[0]
            tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
            compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()
            # print('idx: '+str(idx))
            # print(len(in_channel_idx))
            # print(len(out_channel_idx))
            # iden = compact_conv.weight.data==loose_conv.weight.data
            # print(iden.sum())
            # iden2 = compact_bn.weight.data==loose_bn.weight.data
            # print(iden2.sum())
            # print('-----------')

    for idx in Conv_idx:
        compact_conv = compact_model.module_list[idx][0]
        loose_conv = loose_model.module_list[idx][0]

        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv.weight.data = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.bias.data   = loose_conv.bias.data.clone()


def prune_model_keep_size(model, prune_idx, CBL_idx, CBLidx2mask):

    pruned_model = deepcopy(model)
    for idx in prune_idx:
        mask = torch.from_numpy(CBLidx2mask[idx]).cuda()
        bn_module = pruned_model.module_list[idx][1]

        bn_module.weight.data.mul_(mask)

        activation = F.leaky_relu((1 - mask) * bn_module.bias.data, 0.1)

        # 两个上采样层前的卷积层
        next_idx_list = [idx + 1]
        if idx == 79:
            next_idx_list.append(84)
        elif idx == 91:
            next_idx_list.append(96)

        for next_idx in next_idx_list:
            next_conv = pruned_model.module_list[next_idx][0]
            conv_sum = next_conv.weight.data.sum(dim=(2, 3))
            offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
            if next_idx in CBL_idx:
                next_bn = pruned_model.module_list[next_idx][1]
                next_bn.running_mean.data.sub_(offset)
            else:
                #这里需要注意的是，对于convolutionnal，如果有BN，则该层卷积层不使用bias，如果无BN，则使用bias
                next_conv.bias.data.add_(offset)

        bn_module.bias.data.mul_(mask)

    return pruned_model


def prune_rep_model_keep_size(model, prune_idx, CBL_idx, rep_idx, CBLidx2mask):

    pruned_model = deepcopy(model)
    # for idx in prune_idx:
    #     if idx in rep_idx:
    #         # mask = torch.from_numpy(CBLidx2mask[idx]).cuda()
    #         # conv_module = pruned_model.module_list[idx][0]
    #         # conv_module.weight.data = conv_module.weight.data.permute(1, 2, 3, 0).mul(mask).float().permute(3, 0, 1, 2)
    #         # next_idx_list = [idx + 2]
    #         pass
    #     else:
    #         mask = torch.from_numpy(CBLidx2mask[idx]).cuda()
    #         bn_module = pruned_model.module_list[idx][1]

    #         bn_module.weight.data.mul_(mask)

    #         activation = F.leaky_relu((1 - mask) * bn_module.bias.data, 0.1)

    #         # 两个上采样层前的卷积层
    #         next_idx_list = [idx + 1]
    #         if idx == 60:
    #             next_idx_list.append(65)
    #         elif idx == 72:
    #             next_idx_list.append(77)

    #         for next_idx in next_idx_list:
    #             next_conv = pruned_model.module_list[next_idx][0]
    #             conv_sum = next_conv.weight.data.sum(dim=(2, 3))
    #             offset = conv_sum.matmul(activation.float().reshape(-1, 1)).reshape(-1)
    #             if next_idx in CBL_idx:
    #                 next_bn = pruned_model.module_list[next_idx][1]
    #                 next_bn.running_mean.data.sub_(offset)
    #             else:
    #                 #这里需要注意的是，对于convolutionnal，如果有BN，则该层卷积层不使用bias，如果无BN，则使用bias
    #                 next_conv.bias.data.add_(offset)

    #         bn_module.bias.data.mul_(mask)

    return pruned_model


def obtain_bn_mask(bn_module, thre):

    thre = thre.cuda()
    mask = bn_module.weight.data.abs().ge(thre).float()

    return mask

def obtain_l1_mask(bn_module, random_rate):

    w_copy = bn_module.weight.data.abs().clone()
    w_copy = torch.sum(w_copy, dim=(1,2,3))
    length = w_copy.cpu().numpy().shape[0]
    num_retain = int(length*(1-random_rate))
    _,y = torch.topk(w_copy,num_retain)

    mask = np.zeros(length)
    mask[y.cpu()] = 1

    return mask

def obtain_l1_mask2(bn_module, random_rate):

    w_copy = bn_module.weight.data.abs().clone()
    w_copy = torch.sum(w_copy, dim=(1,2,3))
    length = w_copy.cpu().numpy().shape[0]
    num_retain = int(length*random_rate)
    if num_retain==0:
        num_retain=1
    _,y = torch.topk(w_copy,num_retain)

    mask = np.zeros(length)
    mask[y.cpu()] = 1

    return mask

def obtain_rep_mask(conv_module, distance_rate):
    length = conv_module.weight.data.size()[0]
    codebook = np.ones(length)
    weight_torch = conv_module.weight.data.abs().clone()

    similar_pruned_num = int(weight_torch.size()[0] * distance_rate)
    weight_vec = weight_torch.view(weight_torch.size()[0], -1)
    # norm1 = torch.norm(weight_vec, 1, 1)
    # norm1_np = norm1.cpu().numpy()
    norm2 = torch.norm(weight_vec, 2, 1)
    norm2_np = norm2.cpu().numpy()
    filter_small_index = []
    filter_large_index = []
    filter_large_index = norm2_np.argsort()

    indices = torch.LongTensor(filter_large_index).cuda()
    weight_vec_after_norm = torch.index_select(weight_vec, 0, indices).cpu().numpy()
    # for euclidean distance
    similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'euclidean')
    # for cos similarity
    # similar_matrix = 1 - distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'cosine')
    similar_sum = np.sum(np.abs(similar_matrix), axis=0)

    # for distance similar: get the filter index with largest similarity == small distance
    similar_large_index = similar_sum.argsort()[similar_pruned_num:]
    similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
    similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]

    # kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
    # for x in range(0, len(similar_index_for_filter)):
    #     codebook[
    #     similar_index_for_filter[x] * kernel_length: (similar_index_for_filter[x] + 1) * kernel_length] = 0

    mask = np.ones(length)
    # mask[similar_index_for_filter] = 0

    return mask