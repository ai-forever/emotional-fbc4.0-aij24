import os
import sys

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import numpy as np
from functools import partial
from typing import Optional, Dict, Any

import torch
import transformers
from transformers.utils import logging
from transformers import (AutoModelForCausalLM, AutoTokenizer, AutoConfig)

from team_code.mm_utils import (process_image, process_video, process_audio, tokenizer_multimodal_token,
                      get_model_name_from_path, KeywordsStoppingCriteria)
from team_code.constants import (NUM_FRAMES, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN,
                       DEFAULT_AUDIO_TOKEN, MODAL_INDEX_MAP)

from team_code.projector import load_mm_projector
from team_code.conversation import conv_llama2, conv_llava_llama2
from team_code.videollama2_mistral import Videollama2MistralForCausalLM, Videollama2MistralConfig

logging.set_verbosity("ERROR")


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def setup_model_and_tokenizer(device_map: Optional[str] = "auto",
                              device: Optional[str] = "cuda",
                              use_flash_attn: Optional[bool] = False,
                              **kwargs):
    """
    Load model, tokenizer and processor from checkpoint.
    Note: there is no Internet access!
    """
    model_path = "/app/DAMO-NLP-SG/VideoLLaMA2-7B"
    vision_tower_model_path = "/app/clip-vit-large-patch14-336"

    if not os.path.exists(model_path):
        print(f"Model's checkpoint was not found by path: {model_path}")
        raise FileExistsError(f"Model's checkpoint was not found by path: {model_path}")

    if not os.path.exists(model_path):
        print(f"Model's checkpoint was not found by path: {vision_tower_model_path}")
        raise FileExistsError(f"Model's checkpoint was not found by path: {vision_tower_model_path}")

    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}
    kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = Videollama2MistralForCausalLM.from_pretrained(model_path,
                                                          low_cpu_mem_usage=True,
                                                          config=config,
                                                          **kwargs)
    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(path=vision_tower_model_path)
    vision_tower.to(device=device, dtype=torch.float16)
    # NOTE: model adopts the same processor for processing image and video.
    processor = {
        'image': partial(process_image, processor=vision_tower.image_processor, aspect_ratio=None),
        'video': partial(process_video, processor=vision_tower.image_processor, aspect_ratio=None, num_frames=num_frames),
        'audio': partial(process_audio, processor=None)
    }

    return model, processor, tokenizer

def process_data_sample(sample: Dict[str, Any],
                        processor,
                        modality_type: Optional[str] = 'video',
                        **kwargs) -> Dict[str, Any]:
    """
    Preprocessing function for visual/audio modalities and text instruction.
    :param sample: a data sample;
    Example structure for multiple-choice QA:
        {
            'task_id': 1,
            'task_type': 'qa',
            'question': 'Who is dancing in the end of the video?',
            'video': 'path_to_video.mp4',
            'audio': 'path_to_audio.mp3',
            'choices': [
               {'choice_id': 1, 'choice': 'Woman in red'},
               {'choice_id': 2, 'choice': 'Woman in blue'},
               {'choice_id': 3, 'choice': 'Man in green'},
               {'choice_id': 4, 'choice': 'Man in black'},
               {'choice_id': 5, 'choice': 'Nobody'},
            ] ,
            'correct_answer': 'Woman in blue',
           'correct_answer_choice_id': 2
         }

         or for captioning

        {
          'task_id': 2,
          'task_type': 'captioning',
          'question': 'Describe this video in detail.',
          'video': 'path_to_video.mp4',
          'audio': '',
          'choices': []
        }
    :param processor: processor as a dict of multiple processing functions;
    :param modality_type: a type of requested modality, one from ['image', 'video', 'audio'];
    :param kwargs: addition arguments;
    :return: processed_sample as a Dict[str, Any].
    """
    # 1. Get modality and process it
    modality_path = sample.get(modality_type, None)
    if modality_path is None:
        raise ValueError(f"Modality type: `{modality_type}` was not found in input data sample!")

    if not os.path.exists(modality_path):
        print(f"Error while loading modality (`{modality_type}`) data, can't find a file by path: `{modality_path}`")
        raise FileNotFoundError(f"Error while loading modality (`{modality_type}`) data, can't find a file by path: `{modality_path}`")

    modality_features = processor[modality_type](modality_path)
    if 'return_dtype' in kwargs:
        try:
            return_dtype = getattr(torch, kwargs['return_dtype'])
            modality_features = video_tensor.to(dtype=return_dtype)
        except AttributeError as e:
            print(f"Invalid data type passed: `{kwargs['return_dtype']}`")

    # 2. Construct instruction for task
    if sample['task_type'] == 'qa':
        # Run construction of multiple-choice QA query
        question = sample['question']
        choices = [choice['choice'] for choice in sample['choices']]
        options = ['(A)', '(B)', '(C)', '(D)', '(E)']
        instruction = f"Question: {question}\nOptions:"
        for opt, ans in zip(options, choices):
            instruction += f"\n{opt} {ans}"
        instruction += "\nAnswer with the option\'s letter from the given choices directly and only give the best option."
        return {
            'task_id': sample['task_id'],
            'task_type': sample['task_type'],
            modality_type: modality_features,
            'instruction': instruction,
            'answers': [f"{o} {a}" for o, a in zip(options, choices)]
        }
    elif sample['task_type'] == 'captioning':
        # Run construction of captioning query
        question = sample['question']
        instruction = f"Question: {question}\nAnswer: "
        return {
            'task_id': sample['task_id'],
            'task_type': sample['task_type'],
            modality_type: modality_features,
            'instruction': instruction
        }


def evaluate_generative(sample: Dict[str, Any],
                        model: torch.nn.Module, tokenizer,
                        modality_type: Optional[str] = 'video', **kwargs) -> str:
    """
    Inference function for video-based task answering in generative way.

    Args:
        model: a multimodal model itself;
        sample: a dict-like data sample;
        tokenizer: model's tokenizer;
        modality_type (str): inference modality.
    Returns:
        str: response of the model.
    """

    # 1. text preprocess (tag process & generate prompt).
    if modality_type == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
    elif modality_type == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
    elif modality_type == 'audio':
        modal_token = DEFAULT_AUDIO_TOKEN
    elif modality_type == 'text':
        modal_token = ''
    else:
        raise ValueError(f"Unsupported modality_type: {modality_type}")

    # 1. vision preprocess (load & transform image or video).
    if modality_type == 'text':
        tensor = None
    else:
        modality_features = sample.get(modality_type, None)
        if modality_features is None:
            raise ValueError(f"Modality type: {modality_type} was not found in input data!")
        tensor = modality_features.half().to(model.device)  # as torch tensor
        tensor = [(tensor, modality_type)]

    # 2. text preprocess (tag process & generate prompt).
    question = modal_token + "\n" + sample['instruction']

    conv = conv_llama2.copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_multimodal_token(prompt,
                                           tokenizer,
                                           modal_token,
                                           return_tensors="pt").unsqueeze(0).to(model.device)
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().to(model.device)

    # 3. generate response according to visual signals and prompts.
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images=tensor,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs



def evaluate_ppl(sample: Dict[str, Any],
                 model: torch.nn.Module, tokenizer,
                 modality_type: Optional[str] = 'video', **kwargs) -> int:
    """
    Inference function for multiple choice Video-QA task (i.e. choice selection).
    similar to: https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/dev/interleave/lmms_eval/models/llava_vid.py#L258
    Args:
        model: a multimodal model itself;
        sample: a dict-like data sample;
        tokenizer: model's tokenizer;
        modality_type (str): inference modality.
    Returns:
        str: response of the model.
    """

    # 1. text preprocess (tag process & generate prompt).
    if modality_type == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
    elif modality_type == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
    elif modality_type == 'audio':
        modal_token = DEFAULT_AUDIO_TOKEN
    elif modality_type == 'text':
        modal_token = ''
    else:
        raise ValueError(f"Unsupported modality_type: {modality_type}")

    # 1. vision preprocess (load & transform image or video).
    if modality_type == 'text':
        tensor = None
    else:
        modality_features = sample.get(modality_type, None)
        if modality_features is None:
            raise ValueError(f"Modality type: {modality_type} was not found in input data!")
            
        tensor = modality_features.half().to(model.device)  # as torch tensor
        tensor = [(tensor, modality_type)]

    # 2. text preprocess (tag process & generate prompt).
    question = modal_token + "\n" + sample['instruction']

    # 3. Iterate through options and select the most probable one
    res = []
    options = sample['answers']
    for i, opt in enumerate(options):
        # Tokenize without answer option
        conv = conv_llama2.copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        contxt_id = tokenizer_multimodal_token(prompt,
                                               tokenizer,
                                               modal_token,
                                               return_tensors="pt").unsqueeze(0).to(model.device)
        # Tokenize with answer option
        conv = conv_llama2.copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], f"\nAnswer: {opt}")
        prompt = conv.get_prompt()
        input_ids = tokenizer_multimodal_token(prompt,
                                               tokenizer,
                                               modal_token,
                                               return_tensors="pt").unsqueeze(0).to(model.device)
        attention_masks = input_ids.ne(tokenizer.pad_token_id).long().to(model.device)

        labels = input_ids.clone().to(model.device)

        # Context part no need to calculate for loss
        labels[0, : contxt_id.shape[1] + 3] = -100
        with torch.inference_mode():
            outputs = model(input_ids=input_ids,
                            attention_masks=attention_masks,
                            labels=labels,
                            images=tensor)

        loss = outputs["loss"]
        loss = torch.exp(loss)
        logits = outputs["logits"]
        greedy_tokens = logits.argmax(dim=-1)
        cont_toks = input_ids[:, contxt_id.shape[1]:]  # [1, seq]

        greedy_tokens = greedy_tokens[:, contxt_id.shape[1]: input_ids.shape[1]]  # [1, seq]

        max_equal = (greedy_tokens == cont_toks).all()
        # as tuple: (un-normalized ppl, normalized ppl)
        res.append((float(loss.item()), float((loss / (labels != -100).sum()).item()), bool(max_equal)))

    assert len(res) == len(options)
    predictions_normalized = [pred[1] for pred in res]
    predicted_choice = np.argmin(predictions_normalized)
    return int(predicted_choice)

