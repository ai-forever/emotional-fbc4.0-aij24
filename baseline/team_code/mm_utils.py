import ast
import os
import math
import base64
import traceback
from io import BytesIO
from typing import Optional

import cv2
import torch
import imageio
import torchaudio
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from moviepy.editor import VideoFileClip
from transformers import StoppingCriteria

from team_code.constants import NUM_FRAMES, MAX_FRAMES, NUM_FRAMES_PER_SECOND, MODAL_INDEX_MAP, DEFAULT_IMAGE_TOKEN


def chunk_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def create_photo_grid(arr, rows=None, cols=None):
    """
    Create a photo grid from a 4D numpy array with shape [t, h, w, c].

    Parameters:
        arr (numpy.ndarray): Input array with shape [t, h, w, c].
        rows (int): Optional. Number of rows in the grid. If not set, it will be determined based on `cols` or the square root of `t`.
        cols (int): Optional. Number of columns in the grid. If not set, it will be determined based on `rows` or the square root of `t`.

    Returns:
        numpy.ndarray: A 3D numpy array representing the photo grid.
    """

    if isinstance(arr, list):
        if isinstance(arr[0], Image.Image):
            arr = np.stack([np.array(img) for img in arr])
        elif isinstance(arr[0], np.ndarray):
            arr = np.stack(arr)
        else:
            raise ValueError("Invalid input type. Expected list of Images or numpy arrays.")

    t, h, w, c = arr.shape
    
    # Calculate the number of rows and columns if not provided
    if rows is None and cols is None:
        rows = math.ceil(math.sqrt(t))
        cols = math.ceil(t / rows)
    elif rows is None:
        rows = math.ceil(t / cols)
    elif cols is None:
        cols = math.ceil(t / rows)

    # Check if the grid can hold all the images
    if rows * cols < t:
        raise ValueError(f"Not enough grid cells ({rows}x{cols}) to hold all images ({t}).")
    
    # Create the grid array with appropriate height and width
    grid_height = h * rows
    grid_width = w * cols
    grid = np.zeros((grid_height, grid_width, c), dtype=arr.dtype)
    
    # Fill the grid with images
    for i in range(t):
        row_idx = i // cols
        col_idx = i % cols
        grid[row_idx*h:(row_idx+1)*h, col_idx*w:(col_idx+1)*w, :] = arr[i]
    
    return grid


def process_image(image_path, processor, aspect_ratio='pad'):
    image = Image.open(image_path).convert('RGB')

    images = [np.array(image)]

    if aspect_ratio == 'pad':
        images = [Image.fromarray(f) for f in images]
        images = [expand2square(image, tuple(int(x*255) for x in processor.image_mean)) for image in images]
    else:
        images = [Image.fromarray(f) for f in images]

    images = processor.preprocess(images, return_tensors='pt')['pixel_values']
    return images


def frame_sample(duration, mode='uniform', num_frames=None, fps=None):
    if mode == 'uniform':
        assert num_frames is not None, "Number of frames must be provided for uniform sampling."
        # NOTE: v1 version
        # Calculate the size of each segment from which a frame will be extracted
        seg_size = float(duration - 1) / num_frames

        frame_ids = []
        for i in range(num_frames):
            # Calculate the start and end indices of each segment
            start = seg_size * i
            end   = seg_size * (i + 1)
            # Append the middle index of the segment to the list
            frame_ids.append((start + end) / 2)

        return np.round(np.array(frame_ids) + 1e-6).astype(int)
        # NOTE: v0 version
        # return np.linspace(0, duration-1, num_frames, dtype=int)
    elif mode == 'fps':
        assert fps is not None, "FPS must be provided for FPS sampling."
        segment_len = min(fps // NUM_FRAMES_PER_SECOND, duration)
        return np.arange(segment_len // 2, duration, segment_len, dtype=int)
    else:
        raise ImportError(f'Unsupported frame sampling mode: {mode}')


def process_video(video_path, processor, s=None, e=None, aspect_ratio='pad', num_frames=NUM_FRAMES):
    if isinstance(video_path, str):
        if s is not None and e is not None:
            s = s if s >= 0. else 0.
            e = e if e >= 0. else 0.
            if s > e:
                s, e = e, s
            elif s == e:
                e = s + 1

        # 1. Loading Video
        if os.path.isdir(video_path):                
            frame_files = sorted(os.listdir(video_path))

            fps = 3
            num_frames_of_video = len(frame_files)
        elif video_path.endswith('.gif'):
            gif_reader = imageio.get_reader(video_path)

            fps = 25
            num_frames_of_video = len(gif_reader)
        else:
            vreader = VideoReader(video_path, ctx=cpu(0), num_threads=1)

            fps = vreader.get_avg_fps()
            num_frames_of_video = len(vreader)

        # 2. Determine frame range & Calculate frame indices
        f_start = 0                       if s is None else max(int(s * fps) - 1, 0)
        f_end   = num_frames_of_video - 1 if e is None else min(int(e * fps) - 1, num_frames_of_video - 1)
        frame_indices = list(range(f_start, f_end + 1))

        duration = len(frame_indices)
        # 3. Sampling frame indices 
        if num_frames is None:
            sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='fps', fps=fps)]
        else:
            sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='uniform', num_frames=num_frames)]

        # 4. Acquire frame data
        if os.path.isdir(video_path): 
            video_data = [Image.open(os.path.join(video_path, frame_files[f_idx])) for f_idx in sampled_frame_indices]
        elif video_path.endswith('.gif'):
            video_data = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)) for idx, frame in enumerate(gif_reader) if idx in sampled_frame_indices]
        else:
            video_data = [Image.fromarray(frame) for frame in vreader.get_batch(sampled_frame_indices).asnumpy()]

    elif isinstance(video_path, np.ndarray):
        video_data = [Image.fromarray(f) for f in video_path]
    elif isinstance(video_path, list) and isinstance(video_path[0], np.ndarray):
        video_data = [Image.fromarray(f) for f in video_path]
    elif isinstance(video_path, list) and isinstance(video_path[0], str):
        video_data = [Image.open(f) for f in video_path]
    elif isinstance(video_path, list) and isinstance(video_path[0], Image.Image):
        video_data = video_path
    else:
        raise ValueError(f"Unsupported video path type: {type(video_path)}")

    while num_frames is not None and len(video_data) < num_frames:
        video_data.append(Image.fromarray(np.zeros((*video_data[-1].size, 3), dtype=np.uint8)))

    # MAX_FRAMES filter
    video_data = video_data[:MAX_FRAMES]

    if aspect_ratio == 'pad':
        images = [expand2square(f, tuple(int(x*255) for x in processor.image_mean)) for f in video_data]
        video = processor.preprocess(images, return_tensors='pt')['pixel_values']
    else:
        images = [f for f in video_data]
        video = processor.preprocess(images, return_tensors='pt')['pixel_values']
    return video


def tokenizer_multimodal_token(prompt, tokenizer, multimodal_token=DEFAULT_IMAGE_TOKEN, return_tensors=None):
    """Tokenize text and multimodal tag to input_ids.

    Args:
        prompt (str): Text prompt (w/ multimodal tag), e.g., '<video>\nDescribe the video.'
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer object.
        multimodal_token (int): Token index corresponding to the multimodal tag.
    """
    multimodal_token_index = MODAL_INDEX_MAP.get(multimodal_token, None)
    if multimodal_token_index is None:
        input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    else:
        prompt_chunks = [tokenizer(chunk, add_special_tokens=False).input_ids for idx, chunk in enumerate(prompt.split(multimodal_token))]

        input_ids = []
        for i in range(1, 2 * len(prompt_chunks)):
            if i % 2 == 1:
                input_ids.extend(prompt_chunks[i // 2])
            else:
                input_ids.append(multimodal_token_index)

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


def waveform2melspec(audio_data: torch.Tensor,
                     sample_rate: Optional[float] = 16000,
                     num_mel_bins: Optional[int] = 126,
                     target_length: Optional[int] = 1036,
                     audio_mean: Optional[float] = -4.2677393,
                     audio_std: Optional[float] = 4.5689974):

    # mel shape: (n_mels, T)
    audio_data -= audio_data.mean()
    mel = torchaudio.compliance.kaldi.fbank(
        audio_data,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_length=25,
        frame_shift=10,
    )

    if mel.shape[0] > target_length:
        # split to three parts
        chunk_frames = target_length
        total_frames = mel.shape[0]
        ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
        if len(ranges[1]) == 0:  # if the audio is too short, we just use the first chunk
            ranges[1] = [0]
        if len(ranges[2]) == 0:  # if the audio is too short, we just use the first chunk
            ranges[2] = [0]
        # randomly choose index for each part
        idx_front = np.random.choice(ranges[0])
        idx_middle = np.random.choice(ranges[1])
        idx_back = np.random.choice(ranges[2])

        # select mel
        mel_chunk_front = mel[idx_front:idx_front + chunk_frames, :]
        mel_chunk_middle = mel[idx_middle:idx_middle + chunk_frames, :]
        mel_chunk_back = mel[idx_back:idx_back + chunk_frames, :]
        # stack
        mel_fusion = torch.stack([mel_chunk_front, mel_chunk_middle, mel_chunk_back], dim=0)
    elif mel.shape[0] < target_length:  # padding if too short
        n_repeat = int(target_length / mel.shape[0]) + 1
        mel = mel.repeat(n_repeat, 1)[:target_length, :]
        mel_fusion = torch.stack([mel, mel, mel], dim=0)
    else:  # if equal
        mel_fusion = torch.stack([mel, mel, mel], dim=0)
    mel_fusion = mel_fusion.transpose(1, 2)  # [3, target_length, mel_bins] -> [3, mel_bins, target_length]

    mel_fusion = (mel_fusion - audio_mean) / (audio_std * 2)
    return mel_fusion


def process_audio(audio_path, processor,
                  sample_rate: Optional[float] = 16000,
                  num_mel_bins: Optional[int] = 126,
                  target_length: Optional[int] = 1036,
                  audio_mean: Optional[float] = -4.2677393,
                  audio_std: Optional[float] = 4.5689974) -> torch.Tensor:

    audio_data, origin_sr = torchaudio.load(audio_path)

    if sample_rate != origin_sr:
        # Resample audio
        audio_data = torchaudio.functional.resample(audio_data,
                                                    orig_freq=origin_sr,
                                                    new_freq=sample_rate)
    waveform_melspec = waveform2melspec(audio_data, sample_rate, num_mel_bins,
                                        target_length, audio_mean, audio_std)
    return waveform_melspec



class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)
