import os
from unittest.mock import patch

import torch
from transformers.dynamic_module_utils import get_imports
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

class FlorenceCaption:
    @torch.no_grad()
    def __init__(self, device='cpu'):
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            self.model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True, cache_dir='.')
            self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True, cache_dir='.')
        self.device = device
        self.model.to(device=self.device)

    @torch.no_grad()
    def get_caption_for(self, images, detail_level=1):
        if detail_level <= 0:
            prompt = ['<CAPTION>']*len(images)
        elif detail_level == 1:
            prompt = ['<DETAILED_CAPTION>']*len(images)
        else:
            prompt = ["<MORE_DETAILED_CAPTION>"]*len(images)
        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(device=self.device)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)
        result = []
        for i in range(len(images)):
            parsed_answer = self.processor.post_process_generation(
                generated_text[i], task=prompt[i], image_size=(images[i].width, images[i].height)
            )
            parsed_answer = parsed_answer[prompt[i]].replace('<pad>','')
            result.append(parsed_answer)
        return result       
