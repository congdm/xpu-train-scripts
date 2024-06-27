import time
import torch

from constants import DatasetType

class args:
    device='xpu'
    diffusers_cache_path = '/Data/automatic/models/Diffusers'
    diffusers_pipe_name = 'Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers'
    LORA_NAME = 'Mayu'
    output_folder = f'{LORA_NAME}_lora_{time.time()}'
    tensorboard_logdir = f'./runs/{LORA_NAME}-{time.time()}'

    class dataset:
        type = DatasetType.WAIFUC_LOCAL
        name = 'sakuma_mayu'
        localdir = './mayu_dataset-raw' # if dataset is from local source
        load_dataset_from_cached_files = True

        waifuc_pruned_tags = ['1girl', 'solo'] # tags in this list will be removed from captions
        waifuc_tags_threshold = 0.5 # tags with value < threshold will be removed from captions
        waifuc_prefix_prompt = 'Sakuma Mayu iDOLM@STER Cinderella Girls.'

        hf_image_field = 'image'    # name of image field in hugging face dataset
        hf_text_field = 'text'

        use_florence_caption = False
        florence_use_cpu = True # running florence on XPU can give garbaged results
        florence_batch_size = 4
        florence_print_to_screen = False    # print to screen to check results during processing

    class opt:
        name = 'CAME'
        class CAME:
            lr = 1e-5
            weight_decay = 0.01
            beta1 = 0.9
            beta2 = 0.999
            beta3 = 0.9998
            eps = (1e-30, 1e-16)
        class AdamW:
            lr = 1e-5
            weight_decay = 0.01
        class Adafactor:
            lr = 1e-4
            weight_decay = 0.01
            relative_step=True
            scale_parameter=True
            warmup_init=True
        class SGD:
            lr = 1e-4
            weight_decay = 0.01
            momentum=0.9
            nesterov=True

    resume = None
    epochs = 128
    grad_accu_steps = 2
    batch_size = 1
    ckpt_every_epoch = 1
    gradient_checkpointing = True
    random_flip = False
    uncond_p = 0.30
    uncond_p_t5 = 0.30
    sample_dropout = 0.50

    # Hunyuan Diffusion
    #model = 'DiT-g/2'
    image_size = 1024
    rope_img = 'base512'
    text_states_dim = 1024
    text_states_dim_t5 = 2048
    text_len = 77
    text_len_t5 = 256
    use_fp16 = False
    infer_mode = 'torch'
    use_flash_attn = False
    qk_norm = True
    norm = 'layer'

    pipeline = 'HunyuanPipeline'
    noise_schedule = 'scaled_linear'
    predict_type = 'v_prediction'
    learn_sigma = True
    mse_loss_weight_type = 'min_snr_5' # 'min_snr_5', 'constant'
    beta_start = 0.00085
    beta_end = 0.03
    noise_offset = 0.01

    # pipeline = 'KohyaPipeline'
    # noise_schedule = 'scaled_linear'
    # beta_start = 0.00085
    # beta_end = 0.03
    # noise_offset = 0.0
    # v_parameterization = True
    # loss_type = 'l2'
    # noise_offset_random_strength = False
    # adaptive_noise_scale = None
    # multires_noise_iterations = 6
    # multires_noise_discount = 0.4
    # min_timestep = None
    # max_timestep = None
    # ip_noise_gamma = None
    # ip_noise_gamma_random_strength = None

    # Lora param
    training_parts = 'lora' # lora, lokr
    rank = 16
    alpha = 1
    #target_modules = ['Wqkv', 'q_proj', 'kv_proj', 'out_proj']
    target_modules = ['time_extra_emb.pooler.q_proj', 'to_q', 'to_k', 'to_v', 'to_out.0']
    use_dora = True
    lokr_use_effective_conv2d = True
    lokr_decompose_both = False
    lokr_decompose_factor = 1

    ###
    latents_dtype = torch.float32
    inputs_dtype = torch.float32
    models_dtype = torch.float32
