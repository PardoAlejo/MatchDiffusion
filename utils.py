import pickle
from pathlib import Path

import torch
from torchvision.utils import save_image
from diffusers.utils import export_to_video

import numpy as np


import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

from PIL import Image
def add_args(parser):
    '''
    Add arguments for sampling to a parser
    '''

    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--save_dir", type=str, default='generated_matchcuts', help='Location to samples and metadata')
    parser.add_argument("--prompts", required=True, type=str, nargs='+', help='Prompts to use, corresponding to each view.')
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--reduction", type=str, default='mean')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--noise_level", type=int, default=50, help='Noise level for stage 2')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--save_metadata", action='store_true', help='If true, save metadata about the views. May use lots of disk space, particularly for permutation views.')


    # match diffusion args
    parser.add_argument("--num_joint_steps", type=int, default=10,
                        help='Number of joint steps (Joint Diffusion). (num_inf_steps - num_joint_steps) will be the number of steps for Disjoint Diffusion')
    parser.add_argument("--initial_lambda_1_1", type=float, default=0.5,
                        help='Initial value for lambda_1_1, default 0.5 which will mean equal weight for both paths in the diffusion of path 1')
    parser.add_argument("--final_lambda_1_1", type=float, default=1.0,
                        help='Final value for lambda_1_1, default 1.0 which will mean after num_parallel_steps, all weight will be on path 1 for path 1')
    parser.add_argument("--initial_lambda_2_2", type=float, default=0.5,
                        help='Initial value for lambda_2_2, default 0.5 which will mean equal weight for both paths in the diffusion of path 2')
    parser.add_argument("--final_lambda_2_2", type=float, default=1.0,
                        help='Final value for lambda_2_2, default 1.0 which will mean after num_parallel_steps, all weight will be on path 2 for path 2')
    parser.add_argument("--lambda_schedule", type=str, default='step',
                        help='Type of lambda schedule to use. Options are `step` or `linear_step`')
    
    parser.add_argument("--scheduler", type=str, default='ddim', help='Specific scheduler to user, by default uses the default scheduler for the model')


    return parser


def save_mc(images, sample_dir):
    '''
    Saves the MC images

    images (list of torch.tensor) :
        List of tensors of shape (1,3,H,W) representing the images

    sample_dir (pathlib.Path) :
        pathlib Path object, representing the directory to save to
    '''
    size = images[0].shape[-1]

    for i, image in enumerate(images):
        # comes from original implementation of deepfloyd
        save_name = sample_dir / f'sample_{size}_{i:02}.png'
        save_image(image / 2. + 0.5, save_name, padding=0)

def save_video_mc(videos, sample_dir):
    '''
    Saves the MC images

    images (list of torch.tensor) :
        List of tensors of shape (1,3,H,W) representing the images

    sample_dir (pathlib.Path) :
        pathlib Path object, representing the directory to save to
    '''
    
    # save first 25 of first and second 25 of second
    # TODO: This could be changed depending on user's desired output
    # For now, we are just concatenating the first 25 of the first video and the second 25 of the second video
    trimmed_video = np.concatenate([videos[0,:25,...], videos[1, 25:,...]])
    
    export_to_video(trimmed_video, f"{sample_dir}/joint_video_trimmed.mp4", fps=8)
    for idx, video in enumerate(videos):
        export_to_video(video, f"{sample_dir}/output_{idx}.mp4", fps=8)


def save_metadata(args, save_dir):
    '''
    Saves the following the sample_dir
        1) pickled view object
        2) args for the illusion
    '''

    metadata = {
        'args': args
    }
    with open(save_dir / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
