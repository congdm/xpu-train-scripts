import torch

class args:
    device='xpu'
    diffusers_cache_path = '/Data/automatic/models/Diffusers'
    diffusers_pipe_name = 'Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers'

    dataset_type = 'WAIFUC_LOCAL'
    dataset_name = 'sakuma_mayu'
    dataset_localdir = './mayu_dataset-raw'
    load_dataset_from_cached_files = False
    use_florence_caption = True
    florence_use_cpu = True # running florence on XPU can give garbaged results
    florence_batch_size = 4
    florence_print_to_screen = False    # print to screen to check results during processing

    resume = None #'./lora_1718818657/SakumaMayu2_0003'
    strict = True
    lr = 1e-4
    weight_decay = 0.01
    epochs = 128
    grad_accu_steps = 2
    batch_size = 1
    ckpt_every_epoch = 1
    gradient_checkpointing = True
    uncond_p = 0.44
    uncond_p_t5 = 0.44

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
    noise_offset = 0.1

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
    training_parts = 'lokr' # lora, lokr
    rank = 4
    #target_modules = ['Wqkv', 'q_proj', 'kv_proj', 'out_proj']
    target_modules = ['time_extra_emb.pooler.q_proj', 'to_q', 'to_k', 'to_v', 'to_out.0']
    use_dora = True

    ###
    latents_dtype = torch.float32
    inputs_dtype = torch.float32
    models_dtype = torch.float32

