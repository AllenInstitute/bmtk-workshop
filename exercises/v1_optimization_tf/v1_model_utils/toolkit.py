import os
import json
import numpy as np
import random
import string
import matplotlib.pyplot as plt


def exp_filter(_x, tau_n=5, n=5):
    l = int(tau_n * n)
    kernel = np.exp(-np.arange(l) / tau_n)
    kernel = kernel / np.sum(kernel)
    return np.convolve(_x, kernel)[:-l + 1]


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def apply_style(ax, scale=1, ylabel=.4):
    ax.set_xlabel(ax.get_xlabel(), fontsize=6 * scale)
    ax.set_ylabel(ax.get_ylabel(), fontsize=6 * scale)
    ax.spines['left'].set_linewidth(.5 * scale)
    ax.spines['bottom'].set_linewidth(.5 * scale)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=5 * scale, width=.5 * scale, length=3 * scale)
    ax.yaxis.set_tick_params(labelsize=5 * scale, width=.5 * scale, length=3 * scale)
    ax.yaxis.set_label_coords(-ylabel / 7, 0.5)


def do_inset_colorbar(_ax, _p, _label, loc='right'):
    if loc == 'right':
        bg_pos = [.925, .1, .075, .8]
        in_pos = [.95, .2, .025, .6]
    elif loc == 'left':
        bg_pos = [.025, .1, .15, .8]
        in_pos = [.05, .2, .025, .6]
    elif loc == 'middle':
        bg_pos = [.025 + .5, .1, .15, .8]
        in_pos = [.05 + .5, .2, .025, .6]
    else:
        raise NotImplementedError(f'must implement location {loc}')
    bg_ax = _ax.inset_axes(bg_pos)
    bg_ax.set_xticks([])
    bg_ax.set_yticks([])
    [_a.set_visible(False) for _a in bg_ax.spines.values()]
    inset_ax = _ax.inset_axes(in_pos)
    cbar = plt.colorbar(_p, cax=inset_ax)
    cbar.outline.set_visible(False)
    cbar.ax.set_ylabel(_label)


def get_random_identifier(prefix='', length=4):
    random_identifier = ''.join(
        random.choice(string.ascii_lowercase + string.digits) for _ in range(length))
    sim_name = prefix + random_identifier
    return sim_name

# def split_1(_s):
#      l = []
#      cc = ''
#      for i in range(len(_s)):
#          if _s[i] != ',':
#              cc += _s[i]
#          else:
#              if cc.count('[') > 0 and cc.count(']') == 0:
#                  cc += _s[i]
#              else:
#                  l.append(cc)
#                  cc = ''
#      if cc != '':
#          l.append(cc)
#      return l


# def split_2(_s):
#     l = []
#     if _s.count('[') == 0:
#         return [_s]
#     else:
#         sub = _s[_s.index('[') + 1:_s.index(']')]
#         base = _s[:_s.index('[')]
#         kk = sub.split(',')
#         for tt in kk:
#             if tt.count('-') > 0:
#                 num_a = tt[:tt.index('-')]
#                 num_b = tt[tt.index('-') + 1:]
#                 digit_len = len(num_a)
#                 assert len(num_a) == len(num_b)
#                 l.extend([base + str(a).zfill(digit_len) for a in range(int(tt[:tt.index('-')]), int(tt[tt.index('-') + 1:]) + 1)])
#             else:
#                 l.append(base + tt)
#     return l

# def expand_slurm_nodes(_s):
#     l = []
#     l1 = split_1(_s)
#     for a in l1:
#         l.extend(split_2(a))
#     return l


# def get_tf_config_from_nodelist(node_list, port=12778):
#     cluster = [a + ':' + str(port) for a in node_list]
#     tf_config = dict(cluster=cluster, task=dict(type='worker', index=os.environ.get('SLURM_PROCID', 0)))
#     return json.dumps(tf_config, indent=4)


# def set_tf_config_from_slurm(port=12778):
#     if 'SLURM_JOB_NODELIST' not in os.environ.keys():
#         return 1, 0
#     # print(f'using slurm job nodelist {os.environ["SLURM_JOB_NODELIST"]}')
#     node_list = expand_slurm_nodes(os.environ['SLURM_JOB_NODELIST'])
#     node_list = [a + 'i.juwels' for a in node_list if a.startswith('jwb')]
#     cluster = dict(worker=[a + ':' + str(port) for a in node_list])
#     task_id = int(os.environ['SLURM_PROCID'])
#     new_tf_config = dict(cluster=cluster, task=dict(type='worker', index=task_id))
#     tf_config_str = os.environ.get('TF_CONFIG', '')
#     if tf_config_str == '':
#         tf_config = dict()
#     else:
#         tf_config = json.loads(tf_config_str)
#     for k, v in new_tf_config.items():
#         tf_config[k] = v
#     os.environ['TF_CONFIG'] = json.dumps(tf_config, indent=4)
#     return len(node_list), task_id