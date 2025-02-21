import torch
import torchvision.transforms.functional as F
import numpy as np
import random
import os
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from typing import Any, Dict, Literal, Optional, get_args, Tuple
import json
from transformers import CLIPTokenizer,AutoConfig,AutoTokenizer
# import open_clip

from urllib.request import urlopen
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms,get_tokenizer
from open_clip.factory import HF_HUB_PREFIX,_MODEL_CONFIGS



class ToTensor(object):

    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label)}


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']

        return {'image': F.resize(image, self.size), 'label': F.resize(label, self.size, interpolation=InterpolationMode.BICUBIC)}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.hflip(image), 'label': F.hflip(label)}

        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.vflip(image), 'label': F.vflip(label)}

        return {'image': image, 'label': label}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}




# image_text_mask的代码
TOKENIZER_TYPE = Literal["biomedclip", "clipseg"]
PROMPT_TYPE = Literal[
    "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "random"
]

class FullDataset(Dataset):
    def __init__(self,
                 images_dir: str,
                 masks_dir: str,
                 caps_file:str,
                 tokenizer_type: TOKENIZER_TYPE = "biomedclip",
                 prompt_type: PROMPT_TYPE = "p9",
                 # caps_file: Optional[str] = "./data/TrainDataset/clinicdb_polyp/anns/train.json",
                 img_size: Tuple[int, int] = (352, 352),
                 context_length: int = 77,
                 img_transforms: Optional[transforms.Compose] = None,
                 mask_transforms: Optional[transforms.Compose] = None,
                 override_prompt: Optional[str] = None,
                 zero_prompt: bool = False,
                 data_num: Optional[int | float] = 1.0,
                 ) -> None:
        super().__init__()
        self.prompt_type = prompt_type

        self.img_size = img_size
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_transforms = img_transforms
        self.mask_transforms = mask_transforms
        self.context_length = context_length
        self.data_num = data_num

        # if tokenizer_type in get_args(TOKENIZER_TYPE):
        self.tokenizer_type = tokenizer_type

        if tokenizer_type == "biomedclip":
            self.tokenizer = open_clip.get_tokenizer(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            ).tokenizer

        elif tokenizer_type=="clipseg":  # ie. tokenizer_type == "clipseg":
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "CIDAS/clipseg-rd64-refined"
            )
        else:
            raise TypeError(
                f"tokenizer_type must be one of {get_args(TOKENIZER_TYPE)} but got {tokenizer_type} instead"
            )

        self.zero_prompt = zero_prompt
        self.override_prompt = override_prompt

        with open(caps_file, "r") as fp:
            self.imgs_captions = json.load(fp)
            random.shuffle(self.imgs_captions)

        if type(self.data_num) == float:
            if self.data_num < 0 or self.data_num > 1:
                raise ValueError(
                    f"data_num must be in range [0, 1] but got {self.data_num} instead."
                )
            self.imgs_captions = self.imgs_captions[
                                 : int(len(self.imgs_captions) * self.data_num)
                                 ]
        else:
            self.imgs_captions = self.imgs_captions[: self.data_num]

        # Assign default img_transforms if no img_transforms is passed
        if self.img_transforms is None:
            self.img_transforms = transforms.Compose(
                [
                    transforms.Resize(size=img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

        # Assign default mask_transforms if no mask_transforms is passed
        if self.mask_transforms is None:
            self.mask_transforms = transforms.Compose(
                [
                    transforms.Resize(
                        size=img_size,
                        interpolation=transforms.InterpolationMode.NEAREST_EXACT,
                    ),
                    transforms.ToTensor(),
                ]
            )

    def __len__(self):
        return len(self.imgs_captions)

    def __getitem__(self, index) -> Dict[str, Any]:
        cap = self.imgs_captions[index]

        # Ensure the image is read with RGB channels
        image = Image.open(f"{self.images_dir}/{cap['img_name']}").convert("RGB")
        mask = Image.open(f"{self.masks_dir}/{cap['mask_name']}")

        h, w = mask.height, mask.width

        # Use overrided prompt if provided
        if self.override_prompt:
            prompt = self.override_prompt
        else:
            if self.prompt_type == "random":
                # Randomly select a prompt except the first one i.e., p0
                prompt = random.choice(list(cap["prompts"].values())[1:])
            else:
                prompt = cap["prompts"][self.prompt_type]

            if type(prompt) == list:
                prompt = random.choice(prompt)

        image = self.img_transforms(image)
        mask = self.mask_transforms(mask)[:1]  # ToTensor Gives 3-channeled mask
        text_enc = self.tokenizer(
            text=prompt,
            max_length=self.context_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = text_enc["input_ids"][0]
        attention_mask = text_enc["attention_mask"][0]
        return dict(
            pixel_values=image,
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask=mask,
            mask_name=cap["mask_name"],
            height=h,
            width=w,
            sentence=prompt
        )



class TestDataset:
    def __init__(self, image_root, gt_root, size,
                 caps_file:str,
                 # caps_file: Optional[str] = "./data/TestDataset/TestDataset/clinicdb_polyp/anns/test.json",
                 ):
        self.prompt_type="p5"
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        # self.size = len(self.images)
        self.index = 0
        self.image_root=image_root
        self.gt_root=gt_root


        with open(caps_file, "r") as fp:
            self.imgs_captions = json.load(fp)
            random.shuffle(self.imgs_captions)
        self.size = len(self.imgs_captions)



    def load_data(self):

        cap = self.imgs_captions[self.index]

        image_path = f"{self.image_root}/{cap['img_name']}"

       
        self.images = self.transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
        image=self.images

        name = os.path.basename(image_path)  # 获取文件名

        gt_path = f"{self.gt_root}/{cap['mask_name']}"
        self.gts = Image.open(gt_path).convert("L")
        # self.gts = Image.open(f"{self.gt_root}/{cap['mask_name']}").convert("L")
        gt=self.gts


        prompt = cap["prompts"][self.prompt_type]
        if type(prompt) == list:
            prompt = random.choice(prompt)
 
        self.index += 1

        mask_name=cap['mask_name']


        return image, gt, name, prompt
  

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

