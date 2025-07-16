
import paddle
import torch
from .api_config.log_writer import write_to_log
from .base import APITestBase
import time
from .api_config.config_analyzer import TensorConfig, APIConfig, analyse_configs
from .paddle_to_torch import get_converter

def tensor_numel(tensor_config):
    numel = 1
    for i in tensor_config.shape:
        numel = numel * i
    return numel

def get_tensor_configs(api_config):
    tensor_configs = []
    for arg_config in api_config.args:
        if isinstance(arg_config, TensorConfig):
            tensor_configs.append(arg_config)
        elif isinstance(arg_config, list):
            for j in range(len(arg_config)):
                if isinstance(arg_config[j], TensorConfig):
                    tensor_configs.append(arg_config[j])
        elif isinstance(arg_config, tuple):
            for j in range(len(arg_config)):
                if isinstance(arg_config[j], TensorConfig):
                    tensor_configs.append(arg_config[j])

    for key, arg_config in api_config.kwargs.items():
        if isinstance(arg_config, TensorConfig):
            tensor_configs.append(arg_config)
        elif isinstance(arg_config, list):
            for j in range(len(arg_config)):
                if isinstance(arg_config[j], TensorConfig):
                    tensor_configs.append(arg_config[j])
        elif isinstance(arg_config, tuple):
            for j in range(len(arg_config)):
                if isinstance(arg_config[j], TensorConfig):
                    tensor_configs.append(arg_config[j])
    return tensor_configs

def total_numel(api_config):
    tensor_configs = get_tensor_configs(api_config)
    numel = 0
    for tensor_config in tensor_configs:
        numel = numel + tensor_numel(tensor_config)
    return numel

def print_performance(is_finish, api, config, numel, test_loop, paddle_forward, torch_forward, paddle_backward, torch_backward, is_torch_combined):
    if paddle_forward is not None and torch_forward is not None and paddle_backward is not None and torch_backward is not None:
        print("[Prof]", api, "\t", config, "\t",  numel, "\t", test_loop, "\t", paddle_forward, "\t", torch_forward, "\t", paddle_forward/torch_forward, "\t", paddle_backward, "\t", torch_backward, "\t", paddle_backward/torch_backward, "\t", is_torch_combined, flush=True)
    elif paddle_forward is not None and torch_forward is not None:
        print("[Prof]", api, "\t", config, "\t",  numel, "\t", test_loop, "\t", paddle_forward, "\t", torch_forward, "\t", paddle_forward/torch_forward, "\t", "None", "\t", "None", "\t", "None", "\t", is_torch_combined, flush=True)

api_loop = {'paddle.Tensor.__abs__': 1081430, 'paddle.Tensor.__add__': 4909, 'paddle.Tensor.__and__': 626993, 'paddle.Tensor.__div__': 843500, 'paddle.Tensor.__eq__': 129842, 'paddle.Tensor.__floordiv__': 616075, 'paddle.Tensor.__ge__': 368189, 'paddle.Tensor.__gt__': 40382, 'paddle.Tensor.__le__': 40382, 'paddle.Tensor.__len__': 2108496, 'paddle.Tensor.__lshift__': 1046216, 'paddle.Tensor.__lt__': 286174, 'paddle.Tensor.__matmul__': 6104, 'paddle.Tensor.__mod__': 368371, 'paddle.Tensor.__mul__': 2795, 'paddle.Tensor.__ne__': 38692, 'paddle.Tensor.__neg__': 169274, 'paddle.Tensor.__nonzero__': 308263, 'paddle.Tensor.__or__': 158365, 'paddle.Tensor.__pow__': 89020, 'paddle.Tensor.__radd__': 11905, 'paddle.Tensor.__rlshift__': 152274, 'paddle.Tensor.__rmatmul__': 811721, 'paddle.Tensor.__rmod__': 1058662, 'paddle.Tensor.__rmul__': 65160, 'paddle.Tensor.__ror__': 172078, 'paddle.Tensor.__rpow__': 634258, 'paddle.Tensor.__rrshift__': 154635, 'paddle.Tensor.__rshift__': 1077739, 'paddle.Tensor.__rsub__': 60675, 'paddle.Tensor.__rtruediv__': 41935, 'paddle.Tensor.__rxor__': 171797, 'paddle.Tensor.__sub__': 6993, 'paddle.Tensor.__truediv__': 7542, 'paddle.Tensor.__xor__': 1138100, 'paddle.Tensor.abs': 65200, 'paddle.Tensor.add': 1044111, 'paddle.Tensor.all': 582443, 'paddle.Tensor.amax': 755849, 'paddle.Tensor.amin': 761038, 'paddle.Tensor.any': 524676, 'paddle.Tensor.argmax': 15838, 'paddle.Tensor.astype': 2546, 'paddle.Tensor.atanh': 1155819, 'paddle.Tensor.bmm': 436172, 'paddle.Tensor.broadcast_to': 940740, 'paddle.Tensor.cast': 6064, 'paddle.Tensor.ceil': 1197035, 'paddle.Tensor.chunk': 28949, 'paddle.Tensor.clip': 33823, 'paddle.Tensor.clone': 18935, 'paddle.Tensor.conj': 1288918, 'paddle.Tensor.cos': 380516, 'paddle.Tensor.cumprod': 781350, 'paddle.Tensor.cumsum': 378545, 'paddle.Tensor.detach': 10602268, 'paddle.Tensor.diag_embed': 167722, 'paddle.Tensor.diagonal': 2660899, 'paddle.Tensor.diff': 261307, 'paddle.Tensor.digamma': 1234865, 'paddle.Tensor.dim': 9169733, 'paddle.Tensor.divide': 39354, 'paddle.Tensor.dot': 1138002, 'paddle.Tensor.equal': 135735, 'paddle.Tensor.equal_all': 611615, 'paddle.Tensor.erfinv': 1206010, 'paddle.Tensor.exp': 1190179, 'paddle.Tensor.expand': 17222, 'paddle.Tensor.expand_as': 897065, 'paddle.Tensor.fill_': 648093, 'paddle.Tensor.fill_diagonal_': 455076, 'paddle.Tensor.fill_diagonal_tensor': 457554, 'paddle.Tensor.flatten': 1568196, 'paddle.Tensor.flip': 959896, 'paddle.Tensor.floor': 1149251, 'paddle.Tensor.gather': 1039269, 'paddle.Tensor.gather_nd': 933139, 'paddle.Tensor.imag': 3128161, 'paddle.Tensor.index_select': 1041361, 'paddle.Tensor.inner': 166665, 'paddle.Tensor.inverse': 280118, 'paddle.Tensor.is_complex': 2641751, 'paddle.Tensor.isclose': 797967, 'paddle.Tensor.isinf': 1343931, 'paddle.Tensor.isnan': 1394017, 'paddle.Tensor.item': 503706, 'paddle.Tensor.kthvalue': 467076, 'paddle.Tensor.lerp': 440281, 'paddle.Tensor.less': 1075182, 'paddle.Tensor.lgamma': 1152179, 'paddle.Tensor.log': 1151568, 'paddle.Tensor.log10': 1182570, 'paddle.Tensor.log1p': 1188603, 'paddle.Tensor.logical_and': 1092535, 'paddle.Tensor.logical_not': 1197422, 'paddle.Tensor.logical_or': 1094690, 'paddle.Tensor.logit': 1092594, 'paddle.Tensor.lu': 39242, 'paddle.Tensor.masked_fill': 243090, 'paddle.Tensor.masked_select': 100610, 'paddle.Tensor.matmul': 3273, 'paddle.Tensor.max': 36073, 'paddle.Tensor.mean': 14341, 'paddle.Tensor.min': 9324, 'paddle.Tensor.mm': 762384, 'paddle.Tensor.mod': 1051901, 'paddle.Tensor.mode': 3902, 'paddle.Tensor.moveaxis': 1445413, 'paddle.Tensor.multigammaln': 56833, 'paddle.Tensor.multiply': 547156, 'paddle.Tensor.nansum': 180371, 'paddle.Tensor.neg': 1063949, 'paddle.Tensor.nonzero': 86607, 'paddle.Tensor.norm': 7582, 'paddle.Tensor.not_equal': 1010669, 'paddle.Tensor.outer': 401602, 'paddle.Tensor.pow': 6161, 'paddle.Tensor.prod': 25769, 'paddle.Tensor.quantile': 20524, 'paddle.Tensor.rad2deg': 674321, 'paddle.Tensor.real': 3295264, 'paddle.Tensor.reciprocal': 1150490, 'paddle.Tensor.remainder': 1033639, 'paddle.Tensor.repeat_interleave': 387255, 'paddle.Tensor.reshape': 1726314, 'paddle.Tensor.rot90': 298911, 'paddle.Tensor.round': 1148821, 'paddle.Tensor.rsqrt': 1171727, 'paddle.Tensor.scale': 1105, 'paddle.Tensor.set_': 286084, 'paddle.Tensor.sigmoid': 1161063, 'paddle.Tensor.sign': 1214145, 'paddle.Tensor.signbit': 52918, 'paddle.Tensor.sin': 380358, 'paddle.Tensor.slice_scatter': 670916, 'paddle.Tensor.split': 61809, 'paddle.Tensor.sqrt': 137712, 'paddle.Tensor.square': 1176478, 'paddle.Tensor.squeeze': 1952478, 'paddle.Tensor.std': 32514, 'paddle.Tensor.subtract': 1023336, 'paddle.Tensor.sum': 9074, 'paddle.Tensor.tanh': 1188193, 'paddle.Tensor.tile': 162634, 'paddle.Tensor.tolist': 158675, 'paddle.Tensor.topk': 529775, 'paddle.Tensor.transpose': 2624534, 'paddle.Tensor.tril': 1257215, 'paddle.Tensor.trunc': 1241686, 'paddle.Tensor.unbind': 1179285, 'paddle.Tensor.unflatten': 331346, 'paddle.Tensor.unique': 45095, 'paddle.Tensor.unsqueeze': 2220693, 'paddle.Tensor.var': 2395, 'paddle.Tensor.zero_': 7155, 'paddle.abs': 25654, 'paddle.acos': 1144770, 'paddle.acosh': 1133027, 'paddle.add': 4475, 'paddle.add_n': 109137, 'paddle.addmm': 540141, 'paddle.all': 568466, 'paddle.allclose': 834722, 'paddle.amax': 423649, 'paddle.amin': 423561, 'paddle.angle': 1205464, 'paddle.any': 565025, 'paddle.argmax': 2422, 'paddle.argmin': 909237, 'paddle.argsort': 118625, 'paddle.as_complex': 3212023, 'paddle.as_real': 3170745, 'paddle.as_strided': 576859, 'paddle.asin': 1101486, 'paddle.asinh': 1084332, 'paddle.assign': 3930, 'paddle.atan': 1073676, 'paddle.atan2': 941002, 'paddle.atanh': 1089029, 'paddle.atleast_1d': 3717319, 'paddle.atleast_2d': 2752932, 'paddle.atleast_3d': 2182286, 'paddle.bincount': 166171, 'paddle.bitwise_and': 1119130, 'paddle.bitwise_invert': 1177797, 'paddle.bitwise_left_shift': 1091406, 'paddle.bitwise_not': 1173096, 'paddle.bitwise_or': 1103012, 'paddle.bitwise_right_shift': 1105786, 'paddle.bitwise_xor': 1110069, 'paddle.bmm': 6797, 'paddle.broadcast_tensors': 5789, 'paddle.broadcast_to': 10161, 'paddle.bucketize': 864209, 'paddle.cartesian_prod': 154404, 'paddle.cast': 5170, 'paddle.cdist': 167036, 'paddle.ceil': 1095350, 'paddle.chunk': 636359, 'paddle.clip': 14674, 'paddle.clone': 47105, 'paddle.column_stack': 623842, 'paddle.combinations': 832855, 'paddle.complex': 987329, 'paddle.concat': 3901, 'paddle.conj': 1208025, 'paddle.copysign': 767933, 'paddle.cos': 1122531, 'paddle.cosh': 1104941, 'paddle.count_nonzero': 272799, 'paddle.crop': 501632, 'paddle.cross': 945349, 'paddle.cummax': 426784, 'paddle.cummin': 426816, 'paddle.cumprod': 1108697, 'paddle.cumsum': 302088, 'paddle.deg2rad': 658731, 'paddle.diag': 1091289, 'paddle.diag_embed': 160783, 'paddle.diagflat': 612607, 'paddle.diagonal': 2676121, 'paddle.diagonal_scatter': 322949, 'paddle.diff': 243702, 'paddle.digamma': 1079216, 'paddle.dist': 412370, 'paddle.divide': 861137, 'paddle.dot': 1035204, 'paddle.dstack': 525367, 'paddle.einsum': 10834, 'paddle.equal': 411320, 'paddle.equal_all': 609465, 'paddle.erf': 1159878, 'paddle.erfinv': 1114643, 'paddle.exp': 210648, 'paddle.expand': 903336, 'paddle.expand_as': 112898, 'paddle.expm1': 1141120, 'paddle.eye': 158930, 'paddle.fft.fft': 147315, 'paddle.fft.fft2': 81301, 'paddle.fft.fftn': 19248, 'paddle.fft.fftshift': 220265, 'paddle.fft.hfft': 178823, 'paddle.fft.hfft2': 119158, 'paddle.fft.hfftn': 149664, 'paddle.fft.ifft': 144249, 'paddle.fft.ifft2': 77185, 'paddle.fft.ifftn': 37347, 'paddle.fft.ifftshift': 139474, 'paddle.fft.ihfft': 147903, 'paddle.fft.ihfft2': 106488, 'paddle.fft.ihfftn': 105773, 'paddle.fft.irfft': 142255, 'paddle.fft.irfft2': 14844, 'paddle.fft.irfftn': 14842, 'paddle.fft.rfft': 248298, 'paddle.fft.rfft2': 183097, 'paddle.fft.rfftn': 21196, 'paddle.flatten': 1786109, 'paddle.flip': 200116, 'paddle.floor': 1094992, 'paddle.floor_divide': 1021779, 'paddle.floor_mod': 1003670, 'paddle.fmax': 1016307, 'paddle.fmin': 994715, 'paddle.frac': 441349, 'paddle.full': 63250, 'paddle.full_like': 946534, 'paddle.gammainc': 46705, 'paddle.gammaincc': 47579, 'paddle.gather': 32707, 'paddle.gather_nd': 600860, 'paddle.geometric.segment_max': 278270, 'paddle.geometric.segment_mean': 246654, 'paddle.geometric.segment_min': 297097, 'paddle.geometric.segment_sum': 297265, 'paddle.geometric.send_u_recv': 620816, 'paddle.geometric.send_ue_recv': 102398, 'paddle.geometric.send_uv': 108809, 'paddle.greater_equal': 960158, 'paddle.greater_than': 599378, 'paddle.heaviside': 1030080, 'paddle.histogram': 229242, 'paddle.histogram_bin_edges': 99952, 'paddle.histogramdd': 3731, 'paddle.hstack': 571062, 'paddle.hypot': 269269, 'paddle.i0': 1131975, 'paddle.i0e': 1129536, 'paddle.i1': 1154758, 'paddle.i1e': 1162422, 'paddle.imag': 3253764, 'paddle.increment': 1511227, 'paddle.incubate.nn.functional.fused_matmul_bias': 755928, 'paddle.incubate.segment_max': 264018, 'paddle.incubate.segment_mean': 225392, 'paddle.incubate.segment_min': 262264, 'paddle.incubate.segment_sum': 256766, 'paddle.index_add': 651405, 'paddle.index_fill': 54779, 'paddle.index_put': 45204, 'paddle.index_sample': 46458, 'paddle.index_select': 49942, 'paddle.inner': 166959, 'paddle.is_complex': 1398101, 'paddle.is_empty': 2788443, 'paddle.isclose': 795810, 'paddle.isfinite': 1226465, 'paddle.isin': 39368, 'paddle.isinf': 1215395, 'paddle.isnan': 92437, 'paddle.isneginf': 50296, 'paddle.isposinf': 46813, 'paddle.isreal': 398098, 'paddle.kron': 17055, 'paddle.kthvalue': 321077, 'paddle.lcm': 2439, 'paddle.ldexp': 110799, 'paddle.lerp': 427046, 'paddle.less': 1072518, 'paddle.less_equal': 1017263, 'paddle.less_than': 1075796, 'paddle.lgamma': 1105100, 'paddle.linalg.cholesky': 262973, 'paddle.linalg.cond': 80488, 'paddle.linalg.corrcoef': 29301, 'paddle.linalg.cov': 13137, 'paddle.linalg.det': 378738, 'paddle.linalg.inv': 266252, 'paddle.linalg.lu': 26320, 'paddle.linalg.lu_unpack': 43212, 'paddle.linalg.matrix_norm': 570272, 'paddle.linalg.matrix_power': 69419, 'paddle.linalg.matrix_transpose': 2412360, 'paddle.linalg.multi_dot': 251654, 'paddle.linalg.norm': 517615, 'paddle.linalg.pinv': 7236, 'paddle.linalg.qr': 14657, 'paddle.linalg.slogdet': 408022, 'paddle.linalg.solve': 78366, 'paddle.linalg.svdvals': 21879, 'paddle.linalg.triangular_solve': 359182, 'paddle.linalg.vector_norm': 144081, 'paddle.log': 46314, 'paddle.log10': 1067424, 'paddle.log1p': 653938, 'paddle.log2': 1083735, 'paddle.logaddexp': 85759, 'paddle.logcumsumexp': 650109, 'paddle.logical_and': 210101, 'paddle.logical_not': 317184, 'paddle.logical_or': 1049665, 'paddle.logical_xor': 547278, 'paddle.logit': 1009122, 'paddle.logspace': 111666, 'paddle.logsumexp': 252560, 'paddle.masked_fill': 162582, 'paddle.masked_scatter': 22008, 'paddle.masked_select': 26472, 'paddle.matmul': 5031, 'paddle.matrix_transpose': 2283005, 'paddle.max': 332310, 'paddle.maximum': 123884, 'paddle.mean': 66334, 'paddle.median': 33658, 'paddle.meshgrid': 89869, 'paddle.min': 567765, 'paddle.minimum': 282452, 'paddle.mm': 1109, 'paddle.mod': 997823, 'paddle.mode': 1278, 'paddle.moveaxis': 1317841, 'paddle.multigammaln': 56625, 'paddle.multiplex': 188272, 'paddle.multiply': 16804, 'paddle.mv': 554776, 'paddle.nan_to_num': 67676, 'paddle.nanmean': 76935, 'paddle.nanmedian': 132170, 'paddle.nanquantile': 24523, 'paddle.nansum': 175749, 'paddle.neg': 1013704, 'paddle.negative': 828925, 'paddle.nextafter': 572324, 'paddle.nn.functional.adaptive_avg_pool1d': 112277, 'paddle.nn.functional.adaptive_avg_pool2d': 7020, 'paddle.nn.functional.adaptive_avg_pool3d': 27182, 'paddle.nn.functional.adaptive_log_softmax_with_loss': 7409, 'paddle.nn.functional.adaptive_max_pool1d': 424201, 'paddle.nn.functional.adaptive_max_pool2d': 776020, 'paddle.nn.functional.adaptive_max_pool3d': 759911, 'paddle.nn.functional.affine_grid': 421642, 'paddle.nn.functional.avg_pool2d': 22767, 'paddle.nn.functional.avg_pool3d': 192890, 'paddle.nn.functional.batch_norm': 459308, 'paddle.nn.functional.bilinear': 1129, 'paddle.nn.functional.binary_cross_entropy': 29760, 'paddle.nn.functional.binary_cross_entropy_with_logits': 127109, 'paddle.nn.functional.celu': 1018291, 'paddle.nn.functional.channel_shuffle': 933276, 'paddle.nn.functional.conv1d': 12666, 'paddle.nn.functional.conv1d_transpose': 64462, 'paddle.nn.functional.conv2d': 6667, 'paddle.nn.functional.conv2d_transpose': 8476, 'paddle.nn.functional.conv3d': 107107, 'paddle.nn.functional.conv3d_transpose': 79601, 'paddle.nn.functional.cosine_embedding_loss': 33723, 'paddle.nn.functional.cosine_similarity': 78269, 'paddle.nn.functional.cross_entropy': 1988, 'paddle.nn.functional.elu': 1002721, 'paddle.nn.functional.embedding': 65077, 'paddle.nn.functional.fold': 300156, 'paddle.nn.functional.gelu': 1657, 'paddle.nn.functional.glu': 93494, 'paddle.nn.functional.grid_sample': 170, 'paddle.nn.functional.group_norm': 46629, 'paddle.nn.functional.hardshrink': 1008698, 'paddle.nn.functional.hardsigmoid': 881132, 'paddle.nn.functional.hardswish': 4213, 'paddle.nn.functional.hardtanh': 985886, 'paddle.nn.functional.hinge_embedding_loss': 60144, 'paddle.nn.functional.instance_norm': 307890, 'paddle.nn.functional.interpolate': 22291, 'paddle.nn.functional.l1_loss': 183321, 'paddle.nn.functional.label_smooth': 10028, 'paddle.nn.functional.layer_norm': 534338, 'paddle.nn.functional.leaky_relu': 6453, 'paddle.nn.functional.linear': 1592, 'paddle.nn.functional.local_response_norm': 70993, 'paddle.nn.functional.log_loss': 939713, 'paddle.nn.functional.log_sigmoid': 996038, 'paddle.nn.functional.log_softmax': 58818, 'paddle.nn.functional.lp_pool1d': 302712, 'paddle.nn.functional.lp_pool2d': 548439, 'paddle.nn.functional.margin_ranking_loss': 76657, 'paddle.nn.functional.max_pool2d': 6578, 'paddle.nn.functional.max_pool3d': 51230, 'paddle.nn.functional.max_unpool1d': 270495, 'paddle.nn.functional.max_unpool2d': 166991, 'paddle.nn.functional.max_unpool3d': 335202, 'paddle.nn.functional.mish': 604472, 'paddle.nn.functional.mse_loss': 100009, 'paddle.nn.functional.multi_label_soft_margin_loss': 85118, 'paddle.nn.functional.multi_margin_loss': 42650, 'paddle.nn.functional.nll_loss': 402765, 'paddle.nn.functional.normalize': 183423, 'paddle.nn.functional.npair_loss': 28888, 'paddle.nn.functional.pad': 10279, 'paddle.nn.functional.pairwise_distance': 98918, 'paddle.nn.functional.pixel_shuffle': 19785, 'paddle.nn.functional.pixel_unshuffle': 988474, 'paddle.nn.functional.poisson_nll_loss': 160876, 'paddle.nn.functional.prelu': 64737, 'paddle.nn.functional.relu': 6720, 'paddle.nn.functional.relu6': 5614, 'paddle.nn.functional.rrelu': 982918, 'paddle.nn.functional.selu': 1176675, 'paddle.nn.functional.sequence_mask': 253003, 'paddle.nn.functional.sigmoid': 41875, 'paddle.nn.functional.sigmoid_focal_loss': 27902, 'paddle.nn.functional.silu': 12941, 'paddle.nn.functional.smooth_l1_loss': 323009, 'paddle.nn.functional.soft_margin_loss': 138103, 'paddle.nn.functional.softmax': 5385, 'paddle.nn.functional.softmax_with_cross_entropy': 584052, 'paddle.nn.functional.softplus': 1123791, 'paddle.nn.functional.softshrink': 1094010, 'paddle.nn.functional.softsign': 1155258, 'paddle.nn.functional.square_error_cost': 335915, 'paddle.nn.functional.tanh': 188501, 'paddle.nn.functional.tanhshrink': 1184140, 'paddle.nn.functional.thresholded_relu': 1054707, 'paddle.nn.functional.triplet_margin_with_distance_loss': 16600, 'paddle.nn.functional.unfold': 2174, 'paddle.nn.functional.zeropad2d': 310012, 'paddle.nn.utils.vector_to_parameters': 31448, 'paddle.nonzero': 2134, 'paddle.not_equal': 1152060, 'paddle.numel': 1194101, 'paddle.ones': 2686, 'paddle.ones_like': 137337, 'paddle.outer': 240935, 'paddle.pdist': 67591, 'paddle.polar': 201066, 'paddle.polygamma': 761358, 'paddle.positive': 5553927, 'paddle.pow': 134885, 'paddle.prod': 654735, 'paddle.put_along_axis': 70029, 'paddle.quantile': 16438, 'paddle.rad2deg': 656661, 'paddle.real': 3161647, 'paddle.reciprocal': 211391, 'paddle.reduce_as': 900135, 'paddle.remainder': 1060446, 'paddle.renorm': 428428, 'paddle.repeat_interleave': 154290, 'paddle.reverse': 1064600, 'paddle.roll': 12444, 'paddle.rot90': 295190, 'paddle.round': 1158630, 'paddle.row_stack': 392207, 'paddle.rsqrt': 1183907, 'paddle.scale': 25650, 'paddle.scatter': 84907, 'paddle.scatter_nd': 302506, 'paddle.scatter_nd_add': 9541, 'paddle.searchsorted': 1280503, 'paddle.select_scatter': 520743, 'paddle.sgn': 113508, 'paddle.shape': 2046794, 'paddle.shard_index': 1182788, 'paddle.sign': 1278528, 'paddle.signal.istft': 28515, 'paddle.signal.stft': 57832, 'paddle.Tensor.greater_equal': 989078, 'paddle.Tensor.put_along_axis': 167437, 'paddle.Tensor.rank': 245267, 'paddle.Tensor.slice': 1274860, 'paddle.Tensor.take_along_axis': 275489, 'paddle.cumulative_trapezoid': 25262, 'paddle.empty_like': 815089, 'paddle.incubate.nn.functional.fused_dropout_add': 197722, 'paddle.incubate.nn.functional.fused_rotary_position_embedding': 112170, 'paddle.incubate.softmax_mask_fuse_upper_triangle': 1162369, 'paddle.nn.functional.avg_pool1d': 289540, 'paddle.nn.functional.dropout': 10149143, 'paddle.nn.functional.gather_tree': 849431, 'paddle.nn.functional.margin_cross_entropy': 99748, 'paddle.nn.functional.max_pool1d': 309655, 'paddle.rank': 195863, 'paddle.reshape': 1721419, 'paddle.signbit': 51192, 'paddle.sin': 222853, 'paddle.sinc': 43538, 'paddle.sinh': 1056764, 'paddle.slice_scatter': 673673, 'paddle.sqrt': 129191, 'paddle.square': 215903, 'paddle.squeeze': 1644886, 'paddle.stack': 7413, 'paddle.stanh': 1042537, 'paddle.std': 31462, 'paddle.strided_slice': 1637177, 'paddle.subtract': 167163, 'paddle.sum': 34591, 'paddle.t': 2235928, 'paddle.take': 67543, 'paddle.take_along_axis': 276590, 'paddle.tan': 1035414, 'paddle.tanh': 111168, 'paddle.tensordot': 104857, 'paddle.tile': 837436, 'paddle.tolist': 405578, 'paddle.topk': 14745, 'paddle.trace': 138240, 'paddle.transpose': 2841170, 'paddle.trapezoid': 25802, 'paddle.tril': 343411, 'paddle.triu': 78644, 'paddle.trunc': 1143527, 'paddle.unbind': 1136575, 'paddle.unflatten': 100168, 'paddle.unfold': 603627, 'paddle.unique': 75951, 'paddle.unique_consecutive': 33208, 'paddle.unsqueeze': 2284082, 'paddle.unstack': 802655, 'paddle.vander': 79742, 'paddle.var': 58795, 'paddle.vecdot': 328235, 'paddle.view': 693142, 'paddle.view_as': 726053, 'paddle.vstack': 608330, 'paddle.where': 39634, 'paddle.zeros': 5140, 'paddle.zeros_like': 141544}

class APITestPaddleTorchGPUPerformance(APITestBase):
    def __init__(self, api_config, **kwargs):
        super().__init__(api_config)
        self.test_amp = kwargs.get("test_amp", False)
        self.converter = get_converter()

    def test(self):
        
        if self.need_skip(paddle_only=True):
            print("[Skip]", flush=True)
            return

        if not self.ana_api_info():
            print("ana_api_info failed", flush=True)
            return

        try:
            convert_result = self.converter.convert(self.api_config.api_name)
        except Exception as e:
            print(f"[paddle_to_torch] Convertion failed for {self.api_config.config}: {str(e)}", flush=True)
            write_to_log("paddle_to_torch_failed", self.api_config.config)
            return
        if not convert_result.is_supported:
            print(f"[paddle_to_torch] Unsupported API {self.api_config.api_name}: {convert_result.error_message}", flush=True)
            write_to_log("paddle_to_torch_failed", self.api_config.config)
            return
        if not convert_result.code or not convert_result.code.is_valid():
            print(f"[paddle_to_torch] No code generated for {self.api_config.api_name}", flush=True)
            write_to_log("paddle_to_torch_failed", self.api_config.config)
            return

        numel = total_numel(self.api_config)
        if self.api_config.api_name in api_loop:
            test_loop = api_loop[self.api_config.api_name]
        else:
            test_loop = 2147483647 * 20 // numel
            test_loop = 100000 if test_loop > 100000 else test_loop
        combined = ""
        paddle_forward = None
        torch_forward = None
        paddle_backward = None
        torch_backward = None

        try:
            if not self.gen_numpy_input():
                print("gen_numpy_input failed")
                return
        except Exception as err:
            print("[numpy error]", self.api_config.config, "\n", str(err))
            write_to_log("numpy_error", self.api_config.config)
            return

        try:
            if not self.gen_paddle_input():
                print("gen_paddle_input failed", flush=True)
                return  

            if self.test_amp:
                with paddle.amp.auto_cast():
                    paddle_output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
            else:
                paddle_output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)

            with paddle.no_grad():
                if self.test_amp:
                    with paddle.amp.auto_cast():
                        paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
                        start = time.time()
                        for i in range(test_loop):
                            self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
                        paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
                        end = time.time()
                        paddle_forward = end - start
                else:
                    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
                    start = time.time()
                    for i in range(test_loop):
                        self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
                    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
                    end = time.time()
                    paddle_forward = end - start
        except Exception as err:
            paddle_output = None
            result_outputs = None
            result_outputs_grads = None
            print_performance(False, self.api_config.api_name, self.api_config.config, numel, test_loop, paddle_forward, torch_forward, paddle_backward, torch_backward, combined)
            if "CUDA error" in str(err) or "memory corruption" in str(err):
                raise err
            if "CUDA out of memory" in str(err) or "Out of memory error" in str(err):
                raise err

        try:
            if self.need_check_grad():
                inputs_list = self.get_paddle_input_list()
                result_outputs, result_outputs_grads = self.gen_paddle_output_and_output_grad(paddle_output)
                if len(inputs_list) != 0 and len(result_outputs) != 0 and len(result_outputs_grads) != 0:
                    paddle.grad(result_outputs, inputs_list, grad_outputs=result_outputs_grads,allow_unused=True)
                    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
                    start = time.time()
                    for i in range(test_loop):
                        paddle.grad(result_outputs, inputs_list, grad_outputs=result_outputs_grads,allow_unused=True)
                    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
                    end = time.time()
                    paddle_backward = end - start
        except Exception as err:
            paddle_output = None
            result_outputs = None
            result_outputs_grads = None
            print_performance(False, self.api_config.api_name, self.api_config.config, numel, test_loop, paddle_forward, torch_forward, paddle_backward, torch_backward, combined)
            if "CUDA error" in str(err) or "memory corruption" in str(err):
                raise err
            if "CUDA out of memory" in str(err) or "Out of memory error" in str(err):
                raise err

            return

        paddle_output = None
        result_outputs = None
        result_outputs_grads = None

        try:
            device = torch.device("cuda:0")
            torch.set_default_device(device)
            if not self.gen_torch_input():
                print("gen_torch_input failed", flush=True)
                return

            # torch_args 与 torch_kwargs 是尚未映射的 torch 参数（即按 paddle 的参数顺序与关键字排列的 torch tensors）
            # (弃用)以下代码等价于:
            # torch_output = Paddle2TorchConverter.execute(convert_result, self.torch_args, self.torch_kwargs)
            # 准备执行环境，将参数(torch tensors)直接映射至locals()
            exec_globals = {"torch": torch}
            exec_locals = {
                "args": self.torch_args,
                "kwargs": self.torch_kwargs,
                "result": None,
                **self.torch_kwargs,
            }
            if self.api_config.api_name == "paddle.nn.functional.rnnt_loss":
                if paddle.device.get_device() == "cpu":
                    exec_locals["fused_log_softmax"] = False

            # convert_result.is_torch_corresponding 为 True 时代表有对应的 Torch API
            # 执行 *_compiled 编译好的代码速度更快，定位 compile error 时可删去 _compiled
            code = convert_result.code
            if code.preprocess_compiled:
                exec(code.preprocess_compiled, exec_globals, exec_locals)

            if not convert_result.is_torch_corresponding:
                combined = "combined"

            if code.core_compiled:
                if self.test_amp:
                    with torch.autocast(device_type="cuda"):
                        exec(code.core_compiled, exec_globals, exec_locals)
                else:
                    exec(code.core_compiled, exec_globals, exec_locals)

            # if code.postprocess_compiled:
            #     exec(code.postprocess_compiled, exec_globals, exec_locals)
            output_var = convert_result.output_var or "result"
            torch_output = exec_locals[output_var]

            with torch.no_grad():
                if code.core_compiled:
                    if self.test_amp:
                        with torch.autocast(device_type="cuda"):
                            torch.cuda.synchronize()
                            start = time.time()
                            for i in range(test_loop):
                                exec(code.core_compiled, exec_globals, exec_locals)
                            torch.cuda.synchronize()
                            end = time.time()
                            torch_forward = end - start
                    else:
                        torch.cuda.synchronize()
                        start = time.time()
                        for i in range(test_loop):
                            exec(code.core_compiled, exec_globals, exec_locals)
                        torch.cuda.synchronize()
                        end = time.time()
                        torch_forward = end - start

            del exec_globals, exec_locals, output_var, convert_result, code
        except Exception as err:
            print_performance(False, self.api_config.api_name, self.api_config.config, numel, test_loop, paddle_forward, torch_forward, paddle_backward, torch_backward, combined)
            if "CUDA error" in str(err) or "memory corruption" in str(err):
                raise err
            if "CUDA out of memory" in str(err) or "Out of memory error" in str(err):
                raise err
            return

        try:
            if self.need_check_grad():
                inputs_list = self.get_torch_input_list()
                result_outputs, result_outputs_grads = self.gen_torch_output_and_output_grad(torch_output)
                del self.torch_args, self.torch_kwargs
                if inputs_list and result_outputs and result_outputs_grads:
                    torch.autograd.grad(outputs=result_outputs, inputs=inputs_list, grad_outputs=result_outputs_grads, retain_graph=True)
                    torch.cuda.synchronize()
                    start = time.time()
                    for i in range(test_loop):
                        torch.autograd.grad(
                            outputs=result_outputs,
                            inputs=inputs_list,
                            grad_outputs=result_outputs_grads,
                            retain_graph=True
                        )
                    torch.cuda.synchronize()
                    end = time.time()
                    torch_backward = end - start
                del inputs_list, result_outputs, result_outputs_grads, torch_output
            else:
                del self.torch_args, self.torch_kwargs, torch_output
        except Exception as err:
            print_performance(False, self.api_config.api_name, self.api_config.config, numel, test_loop, paddle_forward, torch_forward, paddle_backward, torch_backward, combined)
            print(str(err))
            if "CUDA error" in str(err) or "memory corruption" in str(err):
                raise err
            if "CUDA out of memory" in str(err) or "Out of memory error" in str(err):
                raise err
            return
        print_performance(True, self.api_config.api_name, self.api_config.config, numel, test_loop, paddle_forward, torch_forward, paddle_backward, torch_backward, combined)