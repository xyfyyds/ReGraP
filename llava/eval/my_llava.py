import argparse
import glob
import json
import os
import random
import re
from io import BytesIO

import numpy as np
import requests
import torch

from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER,
                             IMAGE_TOKEN_INDEX)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (get_model_name_from_path, process_images,
                            tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ReGraPDataset_Mixture(Dataset):
    def __init__(
            self,
            data_root,
            set_name,
            tokenizer,
            set="train",
            center_crop=False,
            device="cuda",
            config=None,
            image_processor=None,
            json_path = './training-data',
            flip_p=0.5,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.image_processor = image_processor
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.set_name = set_name
        self.questions = []
        self.images_path = []
        self.answers = []
        self.has_image = []
        self.file_path = ['feature_description']

        for conversation_type in self.file_path:

            f = open(os.path.join(json_path, set_name, f'{conversation_type}.json'))
            data = json.load(f)
            file_names = [x for x in data.keys()]
            for file_name in file_names:

                questions = []
                answers = []
                for conv in data[file_name]:
                    questions.append(conv['Human'])
                    answers.append(conv['AI'])
                self.questions.extend(questions)
                self.answers.extend(answers)

                self.images_path.extend([file_name] * len(answers))
                if conversation_type == 'text-only-conversation':
                    self.has_image.extend([False] * len(answers))
                else:
                    self.has_image.extend([True] * len(answers))

            print(conversation_type, len(self.questions))
        print('Total: ', len(self.questions), len(self.answers), len(self.images_path), len(self.has_image))
        if set == "train":
            self._length = len(self.questions)
        else:
            self._length = self.num_images
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image_path = self.images_path[i]
        images = [Image.open(image_path).convert("RGB")]

        # --- Maybe flip the image...
        images = [self.flip_transform(image) for image in images]
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.config
        )
        example["images"] = images_tensor
        example['query'] = self.questions[i]
        example['answer'] = self.answers[i]
        example['has_image'] = self.has_image[i]
        example['image_sizes'] = image_sizes
        return example


class ReGraPDataset(Dataset):
    def __init__(
            self,
            data_root,
            set_name,
            train_image_paths,
            tokenizer,
            size=512,
            repeats=1,
            flip_p=0.5,
            set="train",
            center_crop=False,
            device="cuda",
            config=None,
            image_processor=None,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.image_processor = image_processor

        self.size = size
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.image_paths = []
        for x in np.unique(train_image_paths):
            self.image_paths.append(x)

        for image_path in self.image_paths:
            if 'laion' in image_path:
                self.image_paths.remove(image_path)

        self.num_images = len(self.image_paths)
        self._length = self.num_images
        self.set_name = set_name
        if set == "train":
            self._length = self.num_images * repeats
        else:
            self._length = self.num_images

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image_path = self.image_paths[i % self.num_images]
        images = [Image.open(image_path).convert("RGB")]
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.config
        )
        example["images"] = images_tensor
        example["query"] = f'Give a short description on <{self.set_name}> in this image.'
        example["answer"] = 'reference'
        example['image_sizes'] = image_sizes
        example['has_image'] = True
        return example


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def get_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    return tokenizer, model, image_processor, context_len


def get_query(args, query, model, sks_system_prompt=None):
    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    model_name = get_model_name_from_path(args.model_path)
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    if sks_system_prompt is not None:
        conv.system = conv.system + " " + sks_system_prompt
    args.conv = conv
    return args


def get_image_tensor(image_files, model, image_processor):
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    return images_tensor, image_sizes


#     return
def eval_model(args, model, images_tensor, image_sizes, image_processor, tokenizer, return_ids=False):
    # Model
    disable_torch_init()
    prompt = args.conv.get_prompt()
    # print(prompt)
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    # with torch.inference_mode():
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # print(outputs)
    if return_ids:
        return outputs, output_ids
    else:
        return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
