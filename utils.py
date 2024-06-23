import torch
import transformers.optimization

import config
from config import args

def load_dataset_hunyuan(data):
    print('\nLoad and process training data.')
    if not config.dataset_args.load_dataset_from_cached_files:
        if config.dataset_args.type == config.DatasetType.WAIFUC_LOCAL:
            data.load_from_waifuc_local(
                config.dataset_args.localdir, config.dataset_args.name,
                config.dataset_args.waifuc_prefix_prompt,
                config.dataset_args.waifuc_pruned_tags,
                config.dataset_args.use_florence_caption,
            )
        elif config.dataset_args.type == config.DatasetType.HUGGING_FACE:
            data.load_from_hf_dataset(
                config.dataset_args.name,
                image_field=config.dataset_args.hf_image_field,
                text_field=config.dataset_args.hf_text_field,
                use_florence=config.dataset_args.use_florence_caption
            )
        else:
            assert False, "not supported dataset type in config"
    if config.dataset_args.load_dataset_from_cached_files:
        print('Load cached dataset...')
        if config.dataset_args.type == config.DatasetType.WAIFUC_LOCAL:
            data.load_from_pt(config.dataset_args.name)
        else:
            data.load_from_pt(config.dataset_args.name.replace('/','_'))
    print('Done!')

def create_optimizer(optimizer_name, model, verbose=True):
    if optimizer_name == 'SGD':
        lr = config.opt_args.lr
        momentum = config.opt_args.momentum
        weight_decay = config.opt_args.weight_decay
        opt = torch.optim.SGD(model.parameters(),
            lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True
        )
        if verbose:
            print(f"SGD Nesterov Optimizer parameters: lr={lr}, momentum={momentum}, weight_decay={weight_decay}")
    elif optimizer_name == 'AdamW':
        lr = config.opt_args.lr
        weight_decay = config.opt_args.weight_decay
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        if verbose:
            print(f"AdamW Optimizer parameters: lr={lr}, weight_decay={weight_decay}")
    elif optimizer_name == 'Adafactor':
        lr = config.opt_args.lr
        weight_decay = config.opt_args.weight_decay
        scale_parameter = config.opt_args.scale_parameter
        relative_step = config.opt_args.relative_step
        warmup_init = config.opt_args.warmup_init
        if relative_step:
            opt = transformers.optimization.Adafactor(model.parameters(),
                scale_parameter=scale_parameter, relative_step=True, warmup_init=warmup_init, lr=None
            )
            if verbose:
                print(f'Using Adafactor optimizer with time-dependent lr, scale_parameter={scale_parameter}, warmup_init={warmup_init}')
        else:
            opt = transformers.optimization.Adafactor(model.parameters(),
                lr=lr, weight_decay=weight_decay,
                scale_parameter=scale_parameter, relative_step=False, warmup_init=False,
            )
            if verbose:
                print(f'Using Adafactor optimizer with parameters: lr={lr}, weight_decay={weight_decay}, scale_parameter={scale_parameter}')
    else:
        assert False, 'not supported optimizer'
    return opt