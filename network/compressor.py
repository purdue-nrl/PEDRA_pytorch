# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:14:03 2021

@author: aparna

from: https://github.com/epfml/ChocoSGD/blob/0557423bded53687c8955fcf46487779cc29ff07/dl_code/pcode/utils/sparsification.py#L86
"""

import math
import numpy as np
import torch
import copy

def get_n_bits(tensor):
    return 8 * tensor.nelement() * tensor.element_size()


"""define some general compressors, e.g., top_k, random_k, sign"""


class SparsificationCompressor(object):
    def get_top_k(self, x, ratio):
        """it will sample the top 1-ratio of the samples."""
        x_data = x.view(-1)
        x_len = x_data.nelement()
        top_k = max(1, int(x_len * (1 - ratio)))

        # get indices and the corresponding values
        if top_k == 1:
            _, selected_indices = torch.max(x_data.abs(), dim=0, keepdim=True)
        else:
            _, selected_indices = torch.topk(
                x_data.abs(), top_k, largest=True, sorted=False
            )
        #print(x.size(), top_k)
        return x_data[selected_indices], selected_indices

    def get_mask(self, flatten_arr, indices):
        mask = torch.zeros_like(flatten_arr)
        mask[indices] = 1

        mask = mask.byte()
        return mask.float(), (~mask).float()

    def get_random_k(self, x, ratio, is_biased=True):
        """it will randomly sample the 1-ratio of the samples."""
        # get tensor size.
        x_data = x.view(-1)
        x_len = x_data.nelement()
        top_k = max(1, int(x_len * (1 - ratio)))

        # random sample the k indices.
        selected_indices = np.random.choice(x_len, top_k, replace=False)
        selected_indices = torch.LongTensor(selected_indices).to(x.device)

        if is_biased:
            return x_data[selected_indices], selected_indices
        else:
            return x_len / top_k * x_data[selected_indices], selected_indices

    def compress(self, arr, op, compress_ratio, is_biased):
        if "top_k" in op:
            values, indices = self.get_top_k(arr, compress_ratio)
        elif "random_k" in op:
            values, indices = self.get_random_k(arr, compress_ratio)
        else:
            raise NotImplementedError

        # n_bits = get_n_bits(values) + get_n_bits(indices)
        return values, indices


class QuantizationCompressor(object):
    def get_qsgd(self, x, s, is_biased=False):
        # s=255 for level=8
        norm = x.norm(p=2)
        level_float = s * x.abs() / norm
        previous_level = torch.floor(level_float)
        is_next_level = (torch.rand_like(x) < (level_float - previous_level)).float()
        new_level = previous_level + is_next_level
        # assert not torch.isnan(is_next_level).any()
        #print('\n',x, new_level/s)

        scale = 1
        if is_biased:
            d = x.nelement()
            scale = 1.0 / (min(d / (s ** 2), math.sqrt(d) / s) + 1.0)
            #print(scale)

        return scale * torch.sign(x) * norm * (new_level / s)


    def compress(self, arr, op, quantize_level, is_biased):
        if quantize_level != 32:
            s = 2 ** quantize_level - 1
            values = self.get_qsgd(arr, s, is_biased)
        else:
            values = arr
        return values

    def uncompress(self, arr):
        return arr



def flatten_tensors(tensors):
    """
    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat

def unflatten(flat, tensor):
  
   offset=0
   numel = tensor.numel()
   output = (flat.narrow(0, offset, numel).view_as(tensor))
   return output

def unflatten_tensors(flat, tensors):
    """
    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
            unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)

def quantize_tensor(out_msg, comp_fn, quantization_level, is_biased = True):
    #print(quantization_level)
    out_msg_comp = copy.deepcopy(out_msg)
    quantized_values = comp_fn.compress(out_msg_comp, None, quantization_level, is_biased)
    #print(out_msg-quantized_values)
    
    return quantized_values

def quantize_layerwise(out_msg, quantization_level, device, is_biased = False):
    comp_fn = QuantizationCompressor()
    quantized_values = []
    
    for param in out_msg:
        # quantize.
        #print(param.size())
        _quantized_values = comp_fn.compress(param, None, quantization_level, is_biased)
        quantized_values.append(_quantized_values.to(device))

    return quantized_values

def sparsify_layerwise(out_msg, compression_ratio, is_biased=True):
    comp_msg  = []
    comp_fn = SparsificationCompressor()

    for param in out_msg:
        #print(param.size())
        ref  = torch.zeros_like(param)
        p    = flatten_tensors(param)
        comp = torch.zeros_like(p)
        values, indices = comp_fn.compress(p, "top_k", compression_ratio, is_biased)
        indices       = indices.type(torch.cuda.LongTensor)
        comp[indices] = values.type(comp.data.dtype)
        layer_msg     = unflatten(comp, ref)
        out_msg.append(layer_msg)
    # clean buffers
    del ref, p, comp
    return comp_msg


        
        
    