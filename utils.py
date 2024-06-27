import argparse
import multiprocessing
import random

import torch
import transformers.optimization
from came_pytorch import CAME

from constants import DatasetType
from config import args
from prepare_dataset import resize_and_random_crop, DatasetHunyuan, TrainingDatasetHunyuan
import florence

p = None

def load_dataset_hunyuan(data):
    print('\nLoad and process training data.')
    if not args.dataset.load_dataset_from_cached_files:
        if args.dataset.type == DatasetType.WAIFUC_LOCAL:
            data.load_from_waifuc_local(
                args.dataset.localdir, args.dataset.name,
                args.dataset.waifuc_prefix_prompt,
                args.dataset.waifuc_pruned_tags,
                args.dataset.waifuc_tags_threshold,
                args.dataset.use_florence_caption,
            )
        elif args.dataset.type == DatasetType.HUGGING_FACE:
            data.load_from_hf_dataset(
                args.dataset.name,
                image_field=args.dataset.hf_image_field,
                text_field=args.dataset.hf_text_field,
                use_florence=args.dataset.use_florence_caption
            )
        else:
            assert False, "not supported dataset type in config"
    if args.dataset.load_dataset_from_cached_files:
        print('Load cached dataset...')
        if args.dataset.type == DatasetType.WAIFUC_LOCAL:
            data.load_from_pt(args.dataset.name)
        else:
            data.load_from_pt(args.dataset.name.replace('/','_'))
    print('Done!')

def create_optimizer(optimizer_name, model, verbose=True):
    if optimizer_name == 'SGD':
        lr = args.opt.SGD.lr
        momentum = args.opt.SGD.momentum
        weight_decay = args.opt.SGD.weight_decay
        nesterov = args.opt.SGD.nesterov
        opt = torch.optim.SGD(model.parameters(),
            lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov
        )
        if verbose:
            print(f"SGD Optimizer parameters: lr={lr}, momentum={momentum}, weight_decay={weight_decay}, nesterov={nesterov}")
    elif optimizer_name == 'AdamW':
        lr = args.opt.AdamW.lr
        weight_decay = args.opt.AdamW.weight_decay
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        if verbose:
            print(f"AdamW Optimizer parameters: lr={lr}, weight_decay={weight_decay}")
    elif optimizer_name == 'Adafactor':
        lr = args.opt.Adafactor.lr
        weight_decay = args.opt.Adafactor.weight_decay
        scale_parameter = args.opt.Adafactor.scale_parameter
        relative_step = args.opt.Adafactor.relative_step
        warmup_init = args.opt.Adafactor.warmup_init
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
    elif optimizer_name == 'CAME':
        lr = args.opt.CAME.lr
        weight_decay = args.opt.CAME.weight_decay
        beta1 = args.opt.CAME.beta1
        beta2 = args.opt.CAME.beta2
        beta3 = args.opt.CAME.beta3
        opt = CAME(
            model.parameters(),
            lr = lr,
            weight_decay = weight_decay,
            betas = (beta1, beta2, beta3),
            eps= args.opt.CAME.eps
        )
        if verbose:
            print(f'Using CAME optimizer parameters: lr={lr}, weight_decay={weight_decay}, betas={(beta1, beta2, beta3)}')
    else:
        assert False, 'not supported optimizer'
    return opt

def FlorenceCaptioningProcess(conn):
    base_dataset: DatasetHunyuan = None
    fl = florence.FlorenceCaption()
    obj: TrainingLoopProlog.Message

    def process_batch(images, items, detail_level, info):
        result = fl.get_caption_for(images, detail_level=detail_level)
        for i in range(len(images)):
            items[i].text = args.dataset.waifuc_prefix_prompt + result[i]
            items[i].text_t5 = args.dataset.waifuc_prefix_prompt + result[i]
            print(items[i].filename, file=info, end='')
            print(result[i], file=info)

    obj = conn.recv()
    while obj is not None:
        if obj.msg == 'base_dataset':
            base_dataset = obj.data
        elif obj.msg == 'do':
            obj.data = []
            for x in base_dataset:
                if random.random() >= args.sample_dropout:
                    item = TrainingDatasetHunyuan.Item()
                    item.image, item.image_meta_size = resize_and_random_crop(
                        x.image, x.original_size, x.target_size
                    )
                    item.filename = x.filename
                    obj.data.append(item)
            with open('florence_progress.txt', 'a+') as info:
                batch = []
                batch_items = []
                i = 0
                for x in obj.data:
                    batch.append(x.image)
                    batch_items.append(x)
                    if len(batch) >= obj.batch_size:
                        print(f'Batch {i+1}', file=info)
                        process_batch(batch, batch_items, obj.detail_level, info)
                        batch.clear()
                        batch_items.clear()
                        i += 1
                if len(batch) > 0:
                    process_batch(batch, batch_items, obj.detail_level, info)
            conn.send(obj)
        obj = conn.recv()

class TrainingLoopProlog:
    class Message:
        def __init__(self):
            self.msg: str
            self.data = None
            self.detail_level: int
            self.batch_size: int

    def __init__(self):
        self.florence_proc = None
        self.conn = None

    def _start_florence(self, dataset):
        print('Starting Florence-2 process...')
        self.conn, conn2 = multiprocessing.Pipe(duplex=True)
        base: DatasetHunyuan = dataset.base_dataset
        self.florence_proc = multiprocessing.Process(
            target=FlorenceCaptioningProcess,
            args=(conn2,)
        )
        self.florence_proc.start()
        obj = TrainingLoopProlog.Message()
        obj.msg = 'base_dataset'
        obj.data = base
        self.conn.send(obj)
        base._clear()
        obj.msg = 'do'
        obj.data = None
        obj.batch_size = args.dataset.florence_batch_size
        obj.detail_level = 2
        self.conn.send(obj)

    def step(self, epoch, dataset: TrainingDatasetHunyuan, encoder):
        if epoch == 0:
            # First epoch is training without captions
            uncond_p = args.uncond_p
            uncond_p_t5 = args.uncond_p_t5         
            args.uncond_p = 1.1
            args.uncond_p_t5 = 1.1
            cur_data = dataset._current_data()
            dataset.populate_data_from_base(cur_data)
            # Start process Florence
            self._start_florence(dataset)
        else:
            if self.florence_proc is None:
                self._start_florence(dataset)
            # From epoch 2, training with florence captions
            print('Waiting for Florence results...')
            obj: TrainingLoopProlog.Message = self.conn.recv()
            assert obj.msg == 'do'
            assert len(obj.data) > 0 
            dataset._data[dataset._current] = obj.data
            
            obj.msg = 'do'
            obj.data = None
            obj.batch_size = args.dataset.florence_batch_size
            obj.detail_level = 2
            self.conn.send(obj)

        print('Encode images to latents...')
        dataset.encode_latents(encoder)
        print('Encode text embeds...')
        dataset.encode_text_embeds(encoder)
        with open('debug.txt', 'a+') as f:
            print(f'Epoch {epoch}', file=f)
            for x in dataset._data[dataset._current]:
                print(x.text, file=f)

        if epoch == 0:
            # restore original args
            args.uncond_p = uncond_p
            args.uncond_p_t5 = uncond_p_t5

    def cleanup(self):
        self.conn.send(None)
        self.florence_proc.join(10.0)
        self.florence_proc.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-folder')
    parser.add_argument('--resume')
    parser.add_argument('--log-dir')
    cmd_args = parser.parse_args()

    if cmd_args.output_folder:
        print(f'Set output folder to: {cmd_args.output_folder}')
        args.output_folder = cmd_args.output_folder
    if cmd_args.resume:
        args.resume = cmd_args.resume
    if cmd_args.log_dir:
        args.tensorboard_logdir = cmd_args.log_dir
        