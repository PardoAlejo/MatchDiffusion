import argparse
from pathlib import Path

import torch
from diffusers import CogVideoXPipeline

from samplers.samplers_cog import sample_match_cog
from utils import add_args, save_metadata, save_video_mc

import gc  # garbage collection

# Parse args
parser = argparse.ArgumentParser()
parser = add_args(parser)
args = parser.parse_args()

# Do admin stuff
save_dir = Path(args.save_dir) / args.name
save_dir.mkdir(exist_ok=True, parents=True)


# initialize SD-1.5 pipeline
cog_pipeline = CogVideoXPipeline.from_pretrained(
                "THUDM/CogVideoX-5b", torch_dtype=torch.float16
                ).to("cuda")

prompt_embeds = []
for p in args.prompts:
    prompt_embeds.append(cog_pipeline.encode_prompt(p,
                                                    device="cuda",
                                                    do_classifier_free_guidance=True,
                                                    num_videos_per_prompt=1,
                                                    ))
prompt_embeds, negative_prompt_embeds = zip(*prompt_embeds)
prompt_embeds = torch.cat(prompt_embeds)
negative_prompt_embeds = torch.cat(negative_prompt_embeds)  # These are just null embeds


# Save metadata
save_metadata(args, save_dir)

# Sample illusions
for i in range(args.num_samples):
    # Admin stuff
    generator = torch.manual_seed(args.seed + i)
    sample_dir = save_dir / f'{args.seed + i:04}'
    sample_dir.mkdir(exist_ok=True, parents=True)

    # Sample 64x64 image
    videos = sample_match_cog(cog_pipeline, 
                           prompt_embeds,
                           negative_prompt_embeds,
                           num_inference_steps=args.num_inference_steps,
                           num_joint_steps=args.num_joint_steps,
                           guidance_scale=args.guidance_scale,
                           generator=generator,
                           scheduler=args.scheduler,
                           initial_lambda_1_1=args.initial_lambda_1_1,
                           final_lambda_1_1=args.final_lambda_1_1,
                           initial_lambda_2_2=args.initial_lambda_2_2,
                           final_lambda_2_2=args.final_lambda_2_2,
                           lambda_schedule=args.lambda_schedule,
                           )
    

    # clear gpu memory to be able to handle large number of samples
    # Clear video-related variables from GPU memory
    save_video_mc(videos, sample_dir)
    del videos
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()