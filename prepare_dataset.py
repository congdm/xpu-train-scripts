import random

import torch
import intel_extension_for_pytorch as ipex
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

from config import args
import florence
device = args.device

def resize_image(image, origin_size, target_size):
    aspect_ratio = float(origin_size[0]) / float(origin_size[1])
    if origin_size[0] < origin_size[1]:
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)
    image = image.resize((new_width, new_height), Image.LANCZOS)
    return image

def random_crop_image(image, target_size):
    new_width = image.width
    new_height = image.height
    if new_width > target_size[0]:
        x_start = random.randint(0, new_width - target_size[0])
        y_start = 0
    else:
        x_start = 0
        y_start = random.randint(0, new_height - target_size[1])
    image_crop = image.crop((x_start, y_start, x_start + target_size[0], y_start + target_size[1]))
    crops_coords_top_left = (x_start, y_start)
    return image_crop, crops_coords_top_left

class Item:
    def __init__(self):
        self.image = None
        self.filename = None
        self.text = None
        self.text_t5 = None
        self.original_size = None
        self.target_size = None

        self.encoder_hidden_states = None
        self.text_embedding_mask = None
        self.encoder_hidden_states_t5 = None
        self.text_embedding_mask_t5 = None
        self.image_meta_size = None

class DatasetHunyuan(torch.utils.data.Dataset):
    @torch.no_grad()
    def __init__(self):
        from HunyuanDiT.IndexKits.index_kits import ResolutionGroup
        ar_buckets = [
            '9:16', '3:4',
            '1:1',
            '4:3', '16:9',
        ]
        self.ratio_list = []
        self.resolutions = ResolutionGroup(args.image_size, align=16, target_ratios=ar_buckets).data
        self.buckets = {}
        for reso in self.resolutions:
            w, h = str(reso).split('x')
            w = int(w)
            h = int(h)
            self.ratio_list.append(w/h)
            self.buckets[str(reso)] = []
        self.empty_str_item = Item()

    def _find_nearest_reso(self, image_size):
        w = image_size[0]
        h = image_size[1]
        img_ar = w / h
        if img_ar <= 1.0:
            i = 0
            while img_ar > self.ratio_list[i] and i < len(self.ratio_list):
                i = i+1
        else:
            i = len(self.ratio_list)-1
            while img_ar < self.ratio_list[i] and i > 0:
                i = i-1
        reso = str(self.resolutions[i])
        w, h = reso.split('x')
        w = int(w)
        h = int(h)
        return (w, h), reso

    @torch.no_grad()
    def _process_item(self, image, text, danbooru_tags = None):
        original_size = image.size
        target_size, reso = self._find_nearest_reso(original_size)
        if target_size != original_size:
            image = resize_image(image, original_size, target_size)
        
        item = Item()
        item.image = image
        item.original_size = original_size
        item.target_size = target_size
        item.text = text
        item.text_t5 = text
        if danbooru_tags is not None:
            item.text = item.text + '. danbooru tag: ' + danbooru_tags
            item.text_t5 = item.text_t5 + '. danbooru tag: ' + danbooru_tags
        self.buckets[reso].append(item)
        return item

    @torch.no_grad()
    def load_from_hf_dataset(self, hf_dataset, image_field='image', text_field='text', use_florence=False):
        print(f'Processing dataset {hf_dataset}...')
        dataset = load_dataset(hf_dataset, split="train")
        idx = 0
        for x in tqdm(dataset):
            item = self._process_item(x[image_field], x[text_field])
            item.filename = f'idx {idx}'
            idx = idx+1
        if use_florence:
            self._create_florence_captions()
        hf_dataset = hf_dataset.replace('/', '_')
        self.save_to_pt(hf_dataset)

    @torch.no_grad()
    def load_from_waifuc_local(self, dataset_dir, dataset_name, prompt_prefix, pruned_tags=[], use_florence=False):
        print(f'Processing dataset {dataset_name}...')
        from waifuc.source import LocalSource
        from waifuc.action import ModeConvertAction
        source = LocalSource(dataset_dir)
        source = source.attach(
            ModeConvertAction(mode='RGB', force_background='white'),
        ) 
        for item in source:
            text = f'{prompt_prefix}'
            if not use_florence:
                danbooru_tags = ''
                if len(item.meta['tags']) > 0:
                    tags = item.meta['tags']
                    for k in tags.keys():
                        #if (tags[k] >= 0.5) and (k not in pruned_tags): text = text + ' ' + k
                        if k not in pruned_tags: danbooru_tags = danbooru_tags + k + ','
                if danbooru_tags != '':
                    danbooru_tags = danbooru_tags.rstrip(',')
                else:
                    danbooru_tags = None
            else:
                danbooru_tags = None
            self_item = self._process_item(item.image, text, danbooru_tags)
            self_item.filename = item.meta['filename']
        if use_florence:
            self._create_florence_captions()
        self.save_to_pt(dataset_name)

    def _create_florence_captions(self):
        def _process_batch(florenceCaption, batch, batch_items):
            results = florenceCaption.get_caption_for(batch, detail_level=2)
            for i in range(len(results)):
                batch_items[i].text = batch_items[i].text + '. ' + results[i]
                batch_items[i].text_t5 = batch_items[i].text_t5 + '. ' + results[i]
                if args.florence_print_to_screen:
                    print('')
                    print(batch_items[i].filename)
                    print(results[i])
        print('Creating captions for dataset with Florence...')
        if args.florence_use_cpu:
            florenceCaption = florence.FlorenceCaption('cpu')
        else:
            florenceCaption = florence.FlorenceCaption(args.device)
        with tqdm(total=self.__len__()) as pbar:
            for reso in self.buckets:
                batch = []
                batch_items = []
                batch_size = args.florence_batch_size
                for item in self.buckets[reso]:
                    if len(batch) >= batch_size:
                        _process_batch(florenceCaption, batch, batch_items)
                        pbar.update(len(batch))
                        batch = []
                        batch_items = []
                    batch_items.append(item)
                    batch.append(item.image)
                if len(batch) > 0:
                    _process_batch(florenceCaption, batch, batch_items)
                    pbar.update(len(batch))

    def _print_captions_to_file(self, prefix):
        with open(f'{prefix}_captions.txt', 'w') as f:
            for reso in self.buckets:
                for item in self.buckets[reso]:
                    print(item.filename, file=f)
                    print(item.text, file=f)
                    print('', file=f)

    @torch.no_grad()
    def save_to_pt(self, prefix):
        self._print_captions_to_file(prefix)
        torch.save(self.buckets, f'{prefix}_cached.pt')

    @torch.no_grad()
    def load_from_pt(self, prefix):
        self.buckets = torch.load(f'{prefix}_cached.pt')

    @torch.no_grad()
    def fill_t5_token_mask(self, fill_tensor, fill_number, setting_length):
        fill_length = setting_length - fill_tensor.shape[1]
        if fill_length > 0:
            fill_tensor = torch.cat((fill_tensor, fill_number * torch.ones(1, fill_length)), dim=1)
        return fill_tensor

    @torch.no_grad()
    def get_text_info_with_encoder_t5(self, description_t5):
        text_tokens_and_mask = self.tokenizer_t5(
            description_t5,
            max_length=self.text_ctx_len_t5,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        text_input_ids_t5=self.fill_t5_token_mask(text_tokens_and_mask["input_ids"], fill_number=1, setting_length=self.text_ctx_len_t5).long()
        attention_mask_t5=self.fill_t5_token_mask(text_tokens_and_mask["attention_mask"], fill_number=0, setting_length=self.text_ctx_len_t5).bool()
        return text_input_ids_t5, attention_mask_t5

    @torch.no_grad()
    def get_text_info_with_encoder(self, description):
        pad_num = 0
        text_inputs = self.tokenizer(
            description,
            padding="max_length",
            max_length=self.text_ctx_len,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids[0]
        attention_mask = text_inputs.attention_mask[0].bool()
        if pad_num > 0:
            attention_mask[1:pad_num + 1] = False
        return text_input_ids, attention_mask
    
    @torch.no_grad()
    def _encode_latents_item(self, item, vae):
        if item.target_size != item.image.size:
            image, crops_coords_top_left = random_crop_image(
                item.image, item.target_size
            )
        else:
            image = item.image
            crops_coords_top_left = (0, 0)
        image = transforms.ToTensor()(image)
        image = image.unsqueeze(0) * 2.0 - 1.0
        image = image.to(device=device, dtype=args.latents_dtype)
        vae_scaling_factor = vae.config.scaling_factor
        latents = vae.encode(image).latent_dist.sample().mul_(vae_scaling_factor)
        item.latents = latents.cpu().squeeze(0)
        
        item.image_meta_size = torch.tensor(
            [
                item.original_size[0], item.original_size[1],
                item.target_size[0], item.target_size[1],
                crops_coords_top_left[0], crops_coords_top_left[1]
            ],
            dtype=args.inputs_dtype
        )
    
    @torch.no_grad()
    def encode_latents(self, vae):
        vae.to(device=device, dtype=args.latents_dtype)
        with tqdm(total=self.__len__()) as pbar:
            for reso in self.buckets:
                bucket = self.buckets[reso]
                for item in bucket:
                    self._encode_latents_item(item, vae)
                    pbar.update(1)
        vae.cpu()

    @torch.no_grad()
    def _encode_text_embeds_item(self, item):
        text_embedding, text_embedding_mask = self.get_text_info_with_encoder(item.text)
        text_embedding_t5, text_embedding_mask_t5 = self.get_text_info_with_encoder_t5(item.text_t5)

        text_embedding = text_embedding.unsqueeze(0).to(device)
        text_embedding_mask = text_embedding_mask.unsqueeze(0).to(device)
        encoder_hidden_states = self.text_encoder(
            text_embedding,
            attention_mask=text_embedding_mask,
        )[0]
        text_embedding_t5 = text_embedding_t5.to(device).squeeze(1)
        text_embedding_mask_t5 = text_embedding_mask_t5.to(device).squeeze(1)
        output_t5 = self.text_encoder_t5(
            input_ids=text_embedding_t5,
            attention_mask=text_embedding_mask_t5,
            output_hidden_states=True
        )
        encoder_hidden_states_t5 = output_t5['hidden_states'][-1].detach()

        item.encoder_hidden_states = encoder_hidden_states.cpu().squeeze(0)
        item.text_embedding_mask = text_embedding_mask.cpu().squeeze(0)
        item.encoder_hidden_states_t5 = encoder_hidden_states_t5.cpu().squeeze(0)
        item.text_embedding_mask_t5 = text_embedding_mask_t5.cpu().squeeze(0)

    @torch.no_grad()
    def encode_text_embeds(self, text_encoder, tokenizer, text_encoder_t5, tokenizer_t5):
        text_encoder.to(device=device)
        text_encoder_t5.to(device=device, dtype=torch.float16)  # T5 Encoder always in FP16
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.text_encoder_t5 = text_encoder_t5
        self.tokenizer_t5 = tokenizer_t5
        self.text_ctx_len_t5 = 256
        self.text_ctx_len = 77
        with tqdm(total=self.__len__()) as pbar:
            for reso in self.buckets:
                bucket = self.buckets[reso]
                for item in bucket:
                    self._encode_text_embeds_item(item)
                    pbar.update(1)
        self.empty_str_item.text = ''
        self.empty_str_item.text_t5 = ''
        self._encode_text_embeds_item(self.empty_str_item)
        text_encoder.cpu()
        text_encoder_t5.cpu()
        del self.text_encoder
        del self.tokenizer
        del self.text_encoder_t5
        del self.tokenizer_t5
    
    @torch.no_grad()
    def organize_into_batches(self):
        self.idx_list = {}
        self.batch_list = []
        self.num_batches = 0
        for reso in self.resolutions:
            reso = str(reso)
            n = len(self.buckets[reso])
            self.idx_list[reso] = list(range(n))
            nb = n // args.batch_size
            if n % args.batch_size != 0:
                nb = nb+1
            for i in range(nb):
                self.batch_list.append((reso, i))
            self.num_batches += nb

    @torch.no_grad()
    def shuffle(self):
        for reso in self.resolutions:
            reso = str(reso)
            random.shuffle(self.idx_list[reso])
        random.shuffle(self.batch_list)

    @torch.no_grad()
    def _append_item_to_batch(self, batch, reso, idx):
        batch.latents.append(self.buckets[reso][idx].latents)
        batch.encoder_hidden_states.append(self.buckets[reso][idx].encoder_hidden_states)
        batch.text_embedding_mask.append(self.buckets[reso][idx].text_embedding_mask)
        batch.encoder_hidden_states_t5.append(self.buckets[reso][idx].encoder_hidden_states_t5)
        batch.text_embedding_mask_t5.append(self.buckets[reso][idx].text_embedding_mask_t5)
        batch.image_meta_size.append(self.buckets[reso][idx].image_meta_size)

    @torch.no_grad()
    def get_batch(self, idx):
        reso, i = self.batch_list[idx]
        batch = lambda: None
        batch.latents = []
        batch.encoder_hidden_states = []
        batch.text_embedding_mask = []
        batch.encoder_hidden_states_t5 = []
        batch.text_embedding_mask_t5 = []
        batch.image_meta_size = []

        bucket_len = len(self.idx_list[reso])
        for j in range(i, i+args.batch_size):
            self._append_item_to_batch(batch, reso, self.idx_list[reso][j % bucket_len])

        batch.latents = torch.stack(batch.latents)
        batch.encoder_hidden_states = torch.stack(batch.encoder_hidden_states)
        batch.text_embedding_mask = torch.stack(batch.text_embedding_mask)
        batch.encoder_hidden_states_t5 = torch.stack(batch.encoder_hidden_states_t5)
        batch.text_embedding_mask_t5 = torch.stack(batch.text_embedding_mask_t5)
        batch.image_meta_size = torch.stack(batch.image_meta_size)

        return (batch.latents,
            batch.encoder_hidden_states, batch.text_embedding_mask,
            batch.encoder_hidden_states_t5, batch.text_embedding_mask_t5,
            batch.image_meta_size, reso
        )

    @torch.no_grad()
    def print_buckets_info(self):
        for k in self.idx_list:
            print(f'{k}:  {len(self.idx_list[k])} images')

    def __len__(self):
        res = 0
        for reso in self.resolutions:
            reso = str(reso)
            res = res + len(self.buckets[reso])
        return res

    def __getitem__(self, idx):
        reso, idx = self.batch_list[idx]
        if random.random() < args.uncond_p:
            encoder_hidden_states = self.empty_str_item.encoder_hidden_states
            text_embedding_mask = self.empty_str_item.text_embedding_mask
        else:
            encoder_hidden_states = self.buckets[reso][idx].encoder_hidden_states
            text_embedding_mask = self.buckets[reso][idx].text_embedding_mask
        if random.random() < args.uncond_p_t5:
            encoder_hidden_states_t5 = self.empty_str_item.encoder_hidden_states_t5
            text_embedding_mask_t5 = self.empty_str_item.text_embedding_mask_t5
        else:
            encoder_hidden_states_t5 = self.buckets[reso][idx].encoder_hidden_states_t5
            text_embedding_mask_t5 = self.buckets[reso][idx].text_embedding_mask_t5
        return (
            self.buckets[reso][idx].latents,
            encoder_hidden_states, text_embedding_mask,
            encoder_hidden_states_t5, text_embedding_mask_t5,
            self.buckets[reso][idx].image_meta_size,
        )