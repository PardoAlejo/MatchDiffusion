from tqdm import tqdm
import torch

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def create_lambda_schedule(total_steps, num_joint, schedule_type='step', 
                           initial_lambda_1_1=0.5, final_lambda_1_1=1.0,
                           initial_lambda_2_2=0.5, final_lambda_2_2=1.0):
    schedule = []

    for step in range(total_steps):
        if schedule_type == 'step':
            # Step function logic
            if step < num_joint:
                lambda_1_1 = initial_lambda_1_1
                lambda_2_2 = initial_lambda_2_2
            else:
                lambda_1_1 = final_lambda_1_1
                lambda_2_2 = final_lambda_2_2

        elif schedule_type == 'linear_step':
            # Linear interpolation starting from step `num_joint`
            if step < num_joint:
                lambda_1_1 = initial_lambda_1_1
                lambda_2_2 = initial_lambda_2_2
            else:
                # Calculate the remaining steps after `num_joint`
                remaining_steps = total_steps - num_joint
                progress = (step - num_joint) / remaining_steps
                lambda_1_1 = initial_lambda_1_1 + progress * (final_lambda_1_1 - initial_lambda_1_1)
                lambda_2_2 = initial_lambda_2_2 + progress * (final_lambda_2_2 - initial_lambda_2_2)

        else:
            raise ValueError(f"Unsupported schedule_type: {schedule_type}")

        # Calculate complementary values
        lambda_1_2 = 1.0 - lambda_1_1
        lambda_2_1 = 1.0 - lambda_2_2

        # Append the lambdas for the current step to the schedule
        schedule.append((lambda_1_1, lambda_2_2, lambda_1_2, lambda_2_1))

    return schedule

@torch.no_grad()
def sample_cog(model,
            positive_prompt_embeds,
            negative_prompt_embeds, 
            num_inference_steps=100,
            guidance_scale=7.0,
            reduction='mean',
            generator=None,
            guidance_rescale=0.0):

    # Params
    num_images_per_prompt = 1
    #device = model.device
    device = torch.device('cuda')   # Sometimes model device is cpu???
    height = model.unet.config.sample_size * model.vae_scale_factor
    width = model.unet.config.sample_size * model.vae_scale_factor
    batch_size = 1      # TODO: Support larger batch sizes, maybe
    num_prompts = positive_prompt_embeds.shape[0]


    # For CFG
    prompt_embeds = torch.cat([negative_prompt_embeds, positive_prompt_embeds])

    # Setup timesteps
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = model.scheduler.timesteps

    # prepare latent variables
    num_channels_latents = model.transformer.config.in_channels
    latents = model.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )


    for i, t in enumerate(tqdm(timesteps)):
        # If solving an inverse problem, then project x_t so
        # that first component matches reference image's first component

        # Duplicate inputs for CFG
        # Model input is: [ neg_0, neg_1, ..., pos_0, pos_1, ... ]
        model_input = torch.cat([latents] * 2)
        model_input = model.scheduler.scale_model_input(model_input, t)
        
        timestep = t.expand(model_input.shape[0])

        # Predict noise estimate
        noise_pred = model.transformer(
            hidden_states=model_input,
            timestep=t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=None,
            return_dict=False,
        )[0]

        # Extract uncond (neg) and cond noise estimates
        # noise_pred_uncond, noise_pred_text --> [2, 6, 64, 64]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        # perform guidance
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Reduce predicted noise
        noise_pred = noise_pred.view(-1, num_prompts, num_channels_latents, noise_pred.shape[-2], noise_pred.shape[-1])
        
        if guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale)
        
        if reduction == 'mean':
            noise_pred = noise_pred.mean(1)
        elif reduction == 'sum':
            # For factorized diffusion
            noise_pred = noise_pred.sum(1)
        elif reduction == 'alternate':
            noise_pred = noise_pred[:,i%num_prompts]
        else:
            raise ValueError('Reduction must be either `mean` or `alternate`')


        # compute the previous noisy sample x_t -> x_t-1
        latents = model.scheduler.step(
            noise_pred, t, latents, return_dict=False
        )[0]

    # decode the latent variables to videos
    video = model.decode_latents(latents)
    video = model.video_processor.postprocess_video(video=video,) 
                                                    # output_type=output_type) #pil default, numpy alternative
        
    return video



@torch.no_grad()
def sample_match_cog(model,
            positive_prompt_embeds,
            negative_prompt_embeds,
            height=480,
            width=720,
            num_frames=49,
            num_inference_steps=100,
            num_joint_steps=50,
            guidance_scale=7.0,
            generator=None,
            guidance_rescale=0.0,
            initial_lambda_1_1=0.5,
            final_lambda_1_1=1.0,
            initial_lambda_2_2=0.5,
            final_lambda_2_2=1.0,
            lambda_schedule='step',
            scheduler=None,
            ):

    # Params
    num_images_per_prompt = 1
    #device = model.device
    device = torch.device('cuda')   # Sometimes model device is cpu???
    height = height
    width = width
    batch_size = 1      # TODO: Support larger batch sizes, maybe
    num_prompts = positive_prompt_embeds.shape[0] 


    # For CFG
    prompt_embeds = torch.cat([negative_prompt_embeds, positive_prompt_embeds]) # num_prompts*2 x T x C 

    # Setup timesteps
    if scheduler:
        assert scheduler == 'ddim' or scheduler=='ddpm', "Only DDPM or DDIM scheduler is supported for now"
        if scheduler == 'ddpm':
            from diffusers import DDPMScheduler
            scheduler = DDPMScheduler.from_pretrained("THUDM/CogVideoX-5b", subfolder="scheduler")
        elif scheduler == 'ddim':
            from diffusers import DDIMScheduler
            scheduler = DDIMScheduler.from_pretrained("THUDM/CogVideoX-5b", subfolder="scheduler")
        model.scheduler = scheduler
    
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = model.scheduler.timesteps

    # prepare latent variables
    num_channels_latents = model.transformer.config.in_channels # 4 for SD1.5
    latents = model.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )

    latents = latents.repeat(2, 1, 1, 1, 1) # 2 x 4 x 64 x 64

    # Process timesteps in joint (joint diffusion)

    lambdas = create_lambda_schedule(
                                    total_steps=num_inference_steps, 
                                    num_joint=num_joint_steps,
                                    initial_lambda_1_1=initial_lambda_1_1,
                                    final_lambda_1_1=final_lambda_1_1,
                                    initial_lambda_2_2=initial_lambda_2_2,
                                    final_lambda_2_2=final_lambda_2_2,
                                    schedule_type=lambda_schedule
                                    ) # [(lambda_1_1, lambda_2_2, lambda_1_2, lambda_2_1), ...]

    for i, t in enumerate(tqdm(timesteps)):
        
        # Duplicate inputs for CFG
        # Model input is: [ neg_0, neg_1, ..., pos_0, pos_1, ... ]
        model_input = torch.cat([latents] * 2) # 4 x 4 x 64 x 64 for SD1.5
        model_input = model.scheduler.scale_model_input(model_input, t)
        
        timestep = t.expand(model_input.shape[0])

        image_rotary_emb = (
            model._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if model.transformer.config.use_rotary_positional_embeddings
            else None
        )
        # Predict noise estimate
        noise_pred = model.transformer(
            hidden_states=model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0] # 4 x 4 x 64 x 64 for SD1.5

        # Extract uncond (neg) and cond noise estimates
        # noise_pred_uncond, noise_pred_text --> [2, 4, 64, 64], [2, 4, 64, 64]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2) 

        # perform guidance
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        if guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale)
        
        lambda_1_1, lambda_2_2, lambda_1_2, lambda_2_1 = lambdas[i]
        
        # combine the noise predictions from both paths, for each pair of lambdas
        
        # # entangle all the noise components of both paths
        noise_pred_path_1 = lambda_1_1 * noise_pred[0] + lambda_1_2 * noise_pred[1]
        noise_pred_path_2 = lambda_2_1 * noise_pred[0] + lambda_2_2 * noise_pred[1]
        
        noise_pred[0] = noise_pred_path_1
        noise_pred[1] = noise_pred_path_2


        # compute the previous noisy sample x_t -> x_t-1
        output = model.scheduler.step(
            noise_pred, t, latents, return_dict=True
        )
        latents = output['prev_sample']
        

    
    # decode the latent variables to videos
    video = model.decode_latents(latents)
    videos = model.video_processor.postprocess_video(video=video, output_type='np')
    

    return videos
