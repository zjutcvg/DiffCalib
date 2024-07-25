import importlib
import torch
import os
from collections import OrderedDict


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'lib.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        print('Failed to f1ind function: %s', func_name)
        raise

def load_ckpt(args, shift_model=None, focal_model=None, scale_model=None):
    """
    Load checkpoint.
    """
    if os.path.isfile(args.load_ckpt):
        print("loading checkpoint %s" % args.load_ckpt)
        checkpoint = torch.load(args.load_ckpt)
        # if args.shift_model_path:
        #     shift_model_checkpoint = torch.load(args.shift_model_path)
        # if args.focal_model_path:
        #     focal_model_checkpoint = torch.load(args.focal_model_path)
        # if args.scale_model_path:
        #     scale_model_checkpoint = torch.load(args.scale_model_path)
        if shift_model is not None:
            shift_model.load_state_dict(strip_prefix_if_present(checkpoint['shift_model'], 'module.'),
                                    strict=True)
            # shift_model.load_state_dict(strip_prefix_if_present(shift_model_checkpoint['model_state_dict'], 'module.shift_model.'),
                                    # strict=True)
            # shift_model.load_state_dict(strip_prefix_if_present(shift_model_checkpoint['model_state_dict'], 'shift_model.'),
            #                         strict=True)
        if focal_model is not None:
            focal_model.load_state_dict(strip_prefix_if_present(checkpoint['focal_model'], 'module.'),
                                    strict=True)
            # focal_model.load_state_dict(strip_prefix_if_present(focal_model_checkpoint['model_state_dict'], 'module.focal_model.'),
            #                         strict=True)
        if scale_model is not None:
            # scale_model.load_state_dict(strip_prefix_if_present(checkpoint['focal_model'], 'module.'),
            #                         strict=True)
            # scale_model.load_state_dict(strip_prefix_if_present(scale_model_checkpoint['model_state_dict'], 'module.scale_model.'),
            #                         strict=True)
            scale_model.load_state_dict(strip_prefix_if_present(scale_model_checkpoint['model_state_dict'], 'scale_model.'),
                                    strict=True)
        # depth_model.load_state_dict(strip_prefix_if_present(checkpoint['depth_model'], "module."),
        #                             strict=True)
        del checkpoint
        torch.cuda.empty_cache()


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict