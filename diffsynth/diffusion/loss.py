from .base_pipeline import BasePipeline
import torch
import torch.nn.functional as F


def _merge_loss_inputs(inputs):
    if isinstance(inputs, (list, tuple)):
        inputs_shared, inputs_posi = inputs[0], inputs[1]
        return {**inputs_shared, **inputs_posi}
    return dict(inputs)


def _per_sample_mse(pred, target):
    loss = (pred.float() - target.float()).pow(2)
    return loss.flatten(1).mean(dim=1)


def _as_loss_tensor(value, like):
    if value is None:
        return None
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=torch.float32, device=like.device)
    else:
        value = value.to(dtype=torch.float32, device=like.device)
    if value.numel() == 1:
        value = value.reshape(1).expand_as(like)
    return value.reshape_as(like).detach()


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    if "lora" in inputs:
        # Image-to-LoRA models need to load lora here.
        pipe.clear_lora(verbose=0)
        pipe.load_lora(pipe.dit, state_dict=inputs["lora"], hotload=True, verbose=0)

    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"]) * inputs.get("noise_scale", 1.0)
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    if "first_frame_latents" in inputs:
        noise_pred = noise_pred[:, :, 1:]
        training_target = training_target[:, :, 1:]
    
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def FlowMatchDPOLoss(
    pipe: BasePipeline,
    chosen_inputs,
    rejected_inputs,
    beta=0.1,
    lambda_sft=0.1,
    reference_free=True,
    ref_chosen_loss=None,
    ref_rejected_loss=None,
    dpo_weight=None,
):
    chosen_inputs = _merge_loss_inputs(chosen_inputs)
    rejected_inputs = _merge_loss_inputs(rejected_inputs)

    chosen_latents = chosen_inputs["input_latents"]
    rejected_latents = rejected_inputs["input_latents"]
    if chosen_latents.shape != rejected_latents.shape:
        raise ValueError(
            "DPO requires chosen and rejected input_latents to have the same shape, "
            f"got {tuple(chosen_latents.shape)} and {tuple(rejected_latents.shape)}."
        )

    max_timestep_boundary = int(chosen_inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(chosen_inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    if max_timestep_boundary <= min_timestep_boundary:
        raise ValueError(
            "Invalid timestep boundary for DPO: "
            f"min={min_timestep_boundary}, max={max_timestep_boundary}."
        )
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(chosen_latents) * chosen_inputs.get("noise_scale", 1.0)
    chosen_inputs["latents"] = pipe.scheduler.add_noise(chosen_latents, noise, timestep)
    rejected_inputs["latents"] = pipe.scheduler.add_noise(rejected_latents, noise, timestep)
    chosen_target = pipe.scheduler.training_target(chosen_latents, noise, timestep)
    rejected_target = pipe.scheduler.training_target(rejected_latents, noise, timestep)

    if "first_frame_latents" in chosen_inputs:
        chosen_inputs["latents"][:, :, 0:1] = chosen_inputs["first_frame_latents"]
    if "first_frame_latents" in rejected_inputs:
        rejected_inputs["latents"][:, :, 0:1] = rejected_inputs["first_frame_latents"]

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    chosen_pred = pipe.model_fn(**models, **chosen_inputs, timestep=timestep)
    rejected_pred = pipe.model_fn(**models, **rejected_inputs, timestep=timestep)

    if "first_frame_latents" in chosen_inputs:
        chosen_pred = chosen_pred[:, :, 1:]
        chosen_target = chosen_target[:, :, 1:]
    if "first_frame_latents" in rejected_inputs:
        rejected_pred = rejected_pred[:, :, 1:]
        rejected_target = rejected_target[:, :, 1:]

    training_weight = pipe.scheduler.training_weight(timestep)
    chosen_loss = _per_sample_mse(chosen_pred, chosen_target) * training_weight
    rejected_loss = _per_sample_mse(rejected_pred, rejected_target) * training_weight
    delta_theta = -chosen_loss + rejected_loss

    if reference_free:
        dpo_logits = beta * delta_theta
    else:
        ref_chosen_loss = _as_loss_tensor(ref_chosen_loss, chosen_loss)
        ref_rejected_loss = _as_loss_tensor(ref_rejected_loss, rejected_loss)
        if ref_chosen_loss is None or ref_rejected_loss is None:
            raise ValueError("Reference DPO requires ref_chosen_loss and ref_rejected_loss.")
        delta_ref = -ref_chosen_loss + ref_rejected_loss
        dpo_logits = beta * (delta_theta - delta_ref)

    dpo_loss = -F.logsigmoid(dpo_logits)
    dpo_weight = _as_loss_tensor(dpo_weight, dpo_loss)
    if dpo_weight is not None:
        dpo_loss = dpo_loss * dpo_weight
    return dpo_loss.mean() + lambda_sft * chosen_loss.mean()


def FlowMatchSFTAudioVideoLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    # video
    noise = torch.randn_like(inputs["input_latents"])
    inputs["video_latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    # audio
    if inputs.get("audio_input_latents") is not None:
        audio_noise = torch.randn_like(inputs["audio_input_latents"])
        inputs["audio_latents"] = pipe.scheduler.add_noise(inputs["audio_input_latents"], audio_noise, timestep)
        training_target_audio = pipe.scheduler.training_target(inputs["audio_input_latents"], audio_noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred, noise_pred_audio = pipe.model_fn(**models, **inputs, timestep=timestep)

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    if inputs.get("audio_input_latents") is not None:
        loss_audio = torch.nn.functional.mse_loss(noise_pred_audio.float(), training_target_audio.float())
        loss_audio = loss_audio * pipe.scheduler.training_weight(timestep)
        loss = loss + loss_audio
    return loss


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            denom = sigma_ - sigma
            denom = torch.sign(denom) * torch.clamp(denom.abs(), min=1e-6)
            target = (latents_ - inputs_shared["latents"]) / denom
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
