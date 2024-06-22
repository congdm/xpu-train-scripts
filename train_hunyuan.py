import time
import gc

import torch
import intel_extension_for_pytorch as ipex
from diffusers.models.transformers import HunyuanDiT2DModel
from diffusers import HunyuanDiTPipeline
import diffusers.utils.peft_utils as peft_utils
from peft import LoraConfig, PeftModel, get_peft_model, LoKrConfig
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers.optimization import Adafactor, AdafactorSchedule

import hijack_hunyuan_training
from config import args
from prepare_dataset import DatasetHunyuan

from HunyuanDiT.hydit.modules.posemb_layers import init_image_posemb
from HunyuanDiT.hydit.diffusion import create_diffusion

device=args.device
diffusers_cache_path = args.diffusers_cache_path
diffusers_pipe_name = args.diffusers_pipe_name

########################################
# Lora output name
LORA_NAME = 'Mayu'

# Load and preprocess dataset
##############################
data = DatasetHunyuan()
print('\nLoad and process training data.')
if args.load_dataset_from_cached_files:
    print('Load cached latents...')
    if args.dataset_type == 'WAIFUC_LOCAL':
        data.load_from_pt(args.dataset_name)
    else:
        data.load_from_pt(args.dataset_name.replace('/','_'))
else:
    if args.dataset_type == 'WAIFUC_LOCAL':
        data.load_from_waifuc_local(
            args.dataset_localdir, args.dataset_name,
            'Sakuma Mayu from THE iDOLM@STER Cinderella Girls', # prefix prompt
            [], # pruned tags
        )
    else:
        data.load_from_hf_dataset(args.dataset_name)
print('Done!')


# Training model
################################
print('\nLoad training model.')
pipe = HunyuanDiTPipeline.from_pretrained(diffusers_pipe_name, cache_dir=diffusers_cache_path)
model = pipe.transformer
freqs_cis_img = init_image_posemb(
    args.rope_img,
    resolutions=data.resolutions,
    patch_size=model.config.patch_size,
    hidden_size=model.config.hidden_size,
    num_heads=model.num_heads,
    log_fn=print,
    rope_real=True,
)
print('')
if args.pipeline == 'HunyuanPipeline':
    print('Use Hunyuan GaussianDiffusion training pipeline.')
    diffusion = create_diffusion(
        noise_schedule=args.noise_schedule,
        predict_type=args.predict_type,
        learn_sigma=args.learn_sigma,
        mse_loss_weight_type=args.mse_loss_weight_type,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        noise_offset=args.noise_offset,
    )
else:
    from kohya_pipeline import PipelineHunyuan
    kohyaPipe = PipelineHunyuan()

if args.training_parts == "lora":
    if (args.resume is not None) and (args.resume != ''):
        print(f'Resume training from {args.resume}')
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True)
    else:
        loraconfig = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            target_modules=args.target_modules,
            use_dora=args.use_dora,
        )
        model = get_peft_model(model, loraconfig)
elif args.training_parts == "lokr":
    if (args.resume is not None) and (args.resume != ''):
        print(f'Resume training from {args.resume}')
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True)
    else:
        loraconfig = LoKrConfig(
            r=args.rank,
            alpha=args.rank,
            target_modules=args.target_modules,
        )
        model = get_peft_model(model, loraconfig)
else:
    assert False, 'Invalid training parts parameter. Must be lora or lokr.'

####################
####################
    
@torch.no_grad()
def prepare_model_inputs(batch):
    (
        latents,
        encoder_hidden_states, text_embedding_mask,
        encoder_hidden_states_t5, text_embedding_mask_t5,
        image_meta_size, #reso
    ) = batch
    latents = latents.to(device, dtype=latents_dtype)

    style = torch.tensor([0], dtype=torch.int32)
    style = style.repeat(args.batch_size)

    # positional embedding
    reso = f'{int(image_meta_size[0][2].item())}x{int(image_meta_size[0][3].item())}'
    cos_cis_img, sin_cis_img = freqs_cis_img[reso]

    # Model conditions
    model_kwargs = dict(
        encoder_hidden_states=encoder_hidden_states.to(device, dtype=inputs_dtype),
        text_embedding_mask=text_embedding_mask.to(device),
        encoder_hidden_states_t5=encoder_hidden_states_t5.to(device, dtype=inputs_dtype),
        text_embedding_mask_t5=text_embedding_mask_t5.to(device),
        image_meta_size=image_meta_size.to(device, dtype=inputs_dtype),
        style=style.to(device),
        image_rotary_emb=(cos_cis_img.to(device, dtype=inputs_dtype), sin_cis_img.to(device, dtype=inputs_dtype)),
    )
    return latents, model_kwargs

def log_xpu_stats(writer, msg, step):
    stats = torch.xpu.memory_stats()
    #writer.add_scalar(f'XPU peak allocated_bytes {msg}', stats['allocated_bytes.all.peak'], step)
    #writer.add_scalar(f'XPU current allocated_bytes {msg}', stats['allocated_bytes.all.current'], step)
    writer.add_scalar(f'XPU peak reserved_bytes {msg}', stats['reserved_bytes.all.peak'], step)
    #writer.add_scalar(f'XPU current reserved_bytes {msg}', stats['reserved_bytes.all.current'], step)
    #writer.add_scalar(f'XPU peak active_bytes {msg}', stats['active_bytes.all.peak'], step)
    #writer.add_scalar(f'XPU current active_bytes {msg}', stats['active_bytes.all.current'], step)
    torch.xpu.reset_peak_memory_stats()

def check_xpu_reserved_memory(threshold):
    if torch.xpu.memory_stats()['reserved_bytes.all.current'] >= threshold:
        gc.collect()
        torch.xpu.empty_cache()

def log_running_time(writer, start_time, name, step):
    now = time.time()
    diff = now-start_time[0]
    writer.add_scalar(name, diff, step)
    start_time[0] = now
    return diff

def log_gradients_in_model(model, writer, step):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            writer.add_histogram(tag + "/grad", value.grad.cpu(), step)

# Begin Training
################################
latents_dtype = args.latents_dtype
inputs_dtype = args.inputs_dtype
models_dtype = args.models_dtype

if args.training_parts == 'lora':
    print(f"Training LoRA: rank {args.rank}; use DoRA = {args.use_dora}")
elif args.training_parts == 'lokr':
    print(f"Training LoKr: rank {args.rank}")
print(f'Model dtype: {models_dtype}')

# print(f"AdamW Optimizer parameters: lr={args.lr}, weight_decay={args.weight_decay}")
# opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-4)

# print(f'Using Adafactor optimizer with parameters: lr={args.lr}, weight_decay={args.weight_decay}')
# opt = Adafactor(model.parameters(), scale_parameter=True, relative_step=False, warmup_init=False, lr=args.lr, weight_decay=args.weight_decay)
# scheduler = torch.optim.lr_scheduler.CyclicLR(opt, args.lr, args.lr*10, step_size_up=250, cycle_momentum=False)

# print('Using Adafactor optimizer with time-dependent lr.')
# opt = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)

# print(f'Using Prodigy optimizer with weight_decay = {args.weight_decay}')
# opt = Prodigy(model.parameters(), lr=1, weight_decay=args.weight_decay, safeguard_warmup=True)

print(f"SGD Nesterov Optimizer parameters: lr={args.lr}, momentum={0.9}, weight_decay={args.weight_decay}")
opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

# print(f"SGD Optimizer parameters: lr={args.lr}")
# opt = torch.optim.SGD(model.parameters(), lr=args.lr)
# scheduler = torch.optim.lr_scheduler.CyclicLR(opt, args.lr, args.lr*10*2, step_size_up=250, cycle_momentum=False)

data.organize_into_batches()
print('Training buckets: ')
data.print_buckets_info()
print('')
assert args.batch_size == 1, 'currently only support batch_size=1'
data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

print('Encode images to latents...')
data.encode_latents(pipe.vae)
gc.collect()
torch.xpu.empty_cache()
print('Encode text embeds...')
data.encode_text_embeds(pipe.text_encoder, pipe.tokenizer, pipe.text_encoder_2, pipe.tokenizer_2)
gc.collect()
torch.xpu.empty_cache()

model.to(device=device, dtype=models_dtype)
model.train()
if models_dtype == torch.float32:
    model, opt = torch.xpu.optimize(model, optimizer=opt, fuse_update_step=True)
elif models_dtype == torch.bfloat16:
    model, opt = torch.xpu.optimize(model, optimizer=opt, dtype=models_dtype)

writer = SummaryWriter()
timestamp = time.time()
train_steps = 0
running_loss = 0.0
torch.xpu.reset_accumulated_memory_stats()
# Training loop
for epoch in range(args.epochs):
    print(f"Beginning epoch {epoch+1}...")
    data.shuffle()
    step = 0
    for batch in tqdm(data_loader, miniters=1):
        step += 1
        train_steps += 1
        
        #latents, model_kwargs = prepare_model_inputs(data.get_batch(batch_idx))
        latents, model_kwargs = prepare_model_inputs(batch)

        start_time = [time.time()]
        if args.pipeline == 'HunyuanPipeline':
            loss_dict = diffusion.training_losses(model, None, latents, model_kwargs)
            loss = loss_dict["loss"].mean() / args.grad_accu_steps
        elif args.pipeline == 'KohyaPipeline':
            loss = kohyaPipe.training_loss(model, latents, model_kwargs)
        else:
            assert False, 'Invalid training pipeline!'
        log_running_time(writer, start_time, 'forward pass time', train_steps * args.batch_size)
        log_xpu_stats(writer, 'after forward pass', train_steps * args.batch_size)

        loss.backward()
        log_running_time(writer, start_time, 'backward pass time', train_steps * args.batch_size)
        log_xpu_stats(writer, 'after backward pass', train_steps * args.batch_size)

        running_loss += loss.item()
        if (step % args.grad_accu_steps == 0) or (step == data.num_batches):
            #log_gradients_in_model(model, writer, train_steps * args.batch_size)
            if running_loss > 1.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            diff = log_running_time(writer, start_time, 'optimizer time', train_steps * args.batch_size)
            if diff > 10:
                gc.collect()
                torch.xpu.empty_cache()
            log_xpu_stats(writer, 'after optimizer step', train_steps * args.batch_size)

            if step % args.grad_accu_steps != 0:
                running_loss = running_loss * args.grad_accu_steps / (step % args.grad_accu_steps)
            writer.add_scalar("Loss over train steps", running_loss, train_steps * args.batch_size)
            running_loss = 0.0
        
        # scheduler.step()
        # writer.add_scalar("LR", scheduler.get_last_lr()[0], train_steps * args.batch_size)
        # writer.add_scalar("LR", opt.param_groups[0]['lr'], train_steps * args.batch_size)

        check_xpu_reserved_memory(threshold=13e+9)

    if (epoch+1) % args.ckpt_every_epoch == 0:
        save_name = f'./lora_{int(timestamp)}/{LORA_NAME}_{epoch+1:04d}'
        model.save_pretrained(save_name)
        torch.save(
            {
                'optimizer_class': opt.__class__.__qualname__,
                'state_dict': opt.state_dict(),
            },
            save_name + '/optimizer.pt'
        )
    model.cpu()
    print('Encode images to latents...')
    data.encode_latents(pipe.vae)
    gc.collect()
    torch.xpu.empty_cache()
    model.to(device=device)

writer.close()

