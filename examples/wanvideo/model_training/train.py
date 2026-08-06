import torch, os, argparse, accelerate, warnings
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        fp8_models=None,
        offload_models=None,
        resume_from_checkpoint=None, remove_prefix_in_ckpt=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        dpo_beta=0.1,
        dpo_lambda_sft=0.1,
        dpo_reference_free=True,
        dpo_ref_loss_key_chosen="ref_loss_chosen",
        dpo_ref_loss_key_rejected="ref_loss_rejected",
        s2v_ref_rope_mode="legacy_time_offset",
        s2v_ref_source_id=1.0,
        s2v_ref_rope_theta=10000.0,
        s2v_ref_time_base=30,
        s2v_ref_time_margin=9,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True

        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        if task.endswith(":train") and tokenizer_path is None:
            tokenizer_config = None
        else:
            tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        if task.endswith(":train") and audio_processor_path in (None, ""):
            audio_processor_config = None
        else:
            audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)
        self.pipe.configure_s2v_ref_rope(
            mode=s2v_ref_rope_mode,
            source_id=s2v_ref_source_id,
            theta=s2v_ref_rope_theta,
            time_base=s2v_ref_time_base,
            time_margin=s2v_ref_time_margin,
        )
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.tiled = tiled
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.fp8_models = fp8_models
        self.task = task
        self.dpo_beta = dpo_beta
        self.dpo_lambda_sft = dpo_lambda_sft
        self.dpo_reference_free = dpo_reference_free
        self.dpo_ref_loss_key_chosen = dpo_ref_loss_key_chosen
        self.dpo_ref_loss_key_rejected = dpo_ref_loss_key_rejected
        self._warned_dpo_missing_input_image = False
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
    @staticmethod
    def has_data_value(value):
        if value is None:
            return False
        if isinstance(value, str):
            return value != ""
        if isinstance(value, (list, tuple)):
            return len(value) > 0
        try:
            return not bool(torch.isnan(torch.as_tensor(value)).item())
        except (TypeError, ValueError, RuntimeError):
            return True

    @staticmethod
    def first_item(value):
        if isinstance(value, (list, tuple)):
            return value[0] if len(value) > 0 else None
        return value

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                input_image = data.get("input_image", None)
                inputs_shared["input_image"] = self.first_item(input_image) if self.has_data_value(input_image) else data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        if inputs_shared.get("framewise_decoding", False):
            # WanToDance global model
            inputs_shared["num_frames"] = 4 * (len(data["video"]) - 1) + 1
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": self.tiled,
            "tile_size": self.tile_size,
            "tile_stride": self.tile_stride,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def get_required_dpo_value(self, data, key):
        value = data.get(key, None)
        if not self.has_data_value(value):
            raise ValueError(f"DPO task requires `{key}` in each metadata item.")
        return value

    def validate_dpo_pair(self, chosen_video, rejected_video):
        if len(chosen_video) != len(rejected_video):
            raise ValueError(
                "DPO requires chosen_video and rejected_video to have the same frame count, "
                f"got {len(chosen_video)} and {len(rejected_video)}."
            )
        for frame_id, (chosen_frame, rejected_frame) in enumerate(zip(chosen_video, rejected_video)):
            if chosen_frame.size != rejected_frame.size:
                raise ValueError(
                    "DPO requires chosen_video and rejected_video frames to share the same size, "
                    f"but frame {frame_id} has {chosen_frame.size} and {rejected_frame.size}."
                )

    def get_dpo_pipeline_inputs(self, data):
        chosen_video = self.get_required_dpo_value(data, "chosen_video")
        rejected_video = self.get_required_dpo_value(data, "rejected_video")
        self.validate_dpo_pair(chosen_video, rejected_video)

        input_image = data.get("input_image", None)
        if self.has_data_value(input_image):
            input_image = self.first_item(input_image)
        else:
            input_image = chosen_video[0]
            if not self._warned_dpo_missing_input_image:
                warnings.warn("DPO input_image is missing. The chosen first frame will be used as the shared reference image.")
                self._warned_dpo_missing_input_image = True

        chosen_data = data.copy()
        rejected_data = data.copy()
        chosen_data["video"] = chosen_video
        rejected_data["video"] = rejected_video
        chosen_data["input_image"] = input_image
        rejected_data["input_image"] = input_image

        dpo_inputs = {
            "chosen": self.get_pipeline_inputs(chosen_data),
            "rejected": self.get_pipeline_inputs(rejected_data),
        }
        if self.has_data_value(data.get(self.dpo_ref_loss_key_chosen, None)):
            dpo_inputs["ref_chosen_loss"] = data[self.dpo_ref_loss_key_chosen]
        if self.has_data_value(data.get(self.dpo_ref_loss_key_rejected, None)):
            dpo_inputs["ref_rejected_loss"] = data[self.dpo_ref_loss_key_rejected]
        if self.has_data_value(data.get("dpo_weight", None)):
            dpo_inputs["dpo_weight"] = data["dpo_weight"]
        elif self.has_data_value(data.get("preference_score_gap", None)):
            dpo_inputs["dpo_weight"] = data["preference_score_gap"]
        if self.has_data_value(data.get("pair_id", None)):
            dpo_inputs["pair_id"] = data["pair_id"]
        return dpo_inputs

    def normalize_dpo_inputs(self, inputs):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            return {"chosen": inputs[0], "rejected": inputs[1]}
        if not isinstance(inputs, dict) or "chosen" not in inputs or "rejected" not in inputs:
            raise ValueError("DPO inputs must contain `chosen` and `rejected` entries.")
        return inputs

    def run_pipeline_units(self, inputs):
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return inputs

    def forward_dpo(self, data, inputs=None):
        dpo_inputs = self.get_dpo_pipeline_inputs(data) if inputs is None else self.normalize_dpo_inputs(inputs)
        dpo_inputs = self.transfer_data_to_device(dpo_inputs, self.pipe.device, self.pipe.torch_dtype)
        dpo_inputs["chosen"] = self.run_pipeline_units(dpo_inputs["chosen"])
        dpo_inputs["rejected"] = self.run_pipeline_units(dpo_inputs["rejected"])

        if self.task.endswith(":data_process"):
            return dpo_inputs

        return FlowMatchDPOLoss(
            self.pipe,
            dpo_inputs["chosen"],
            dpo_inputs["rejected"],
            beta=self.dpo_beta,
            lambda_sft=self.dpo_lambda_sft,
            reference_free=self.dpo_reference_free,
            ref_chosen_loss=dpo_inputs.get("ref_chosen_loss", None),
            ref_rejected_loss=dpo_inputs.get("ref_rejected_loss", None),
            dpo_weight=dpo_inputs.get("dpo_weight", None),
        )

    def forward(self, data, inputs=None):
        if self.task.startswith("dpo"):
            return self.forward_dpo(data, inputs=inputs)
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def parse_int_pair(value):
    if isinstance(value, tuple):
        return value
    parts = str(value).replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected two integers, for example 30,52 or 30x52.")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected two integers, for example 30,52 or 30x52.") from exc


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO beta for preference logits.")
    parser.add_argument("--dpo_lambda_sft", type=float, default=0.1, help="Weight of the chosen SFT regularization term in DPO.")
    parser.set_defaults(dpo_reference_free=True)
    parser.add_argument("--dpo_reference_free", dest="dpo_reference_free", action="store_true", help="Use reference-free DPO. This is the default.")
    parser.add_argument("--dpo_use_reference", dest="dpo_reference_free", action="store_false", help="Use precomputed reference losses for DPO.")
    parser.add_argument("--dpo_ref_loss_key_chosen", type=str, default="ref_loss_chosen", help="Metadata/cache key for chosen reference loss.")
    parser.add_argument("--dpo_ref_loss_key_rejected", type=str, default="ref_loss_rejected", help="Metadata/cache key for rejected reference loss.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--framewise_decoding", default=False, action="store_true", help="Enable it if this model is a WanToDance global model.")
    parser.add_argument("--tiled", default=False, action="store_true", help="Use tiled VAE encode/decode in pipeline preprocessing.")
    parser.add_argument("--tile_size", type=parse_int_pair, default=(30, 52), help="VAE tile size as H,W or HxW. Used with --tiled.")
    parser.add_argument("--tile_stride", type=parse_int_pair, default=(15, 26), help="VAE tile stride as H,W or HxW. Used with --tiled.")
    parser.add_argument("--s2v_ref_rope_mode", type=str, default="legacy_time_offset", choices=("legacy_time_offset", "source_id_time_offset", "source_id_local"), help="Reference-frame RoPE mode for Wan2.2-S2V.")
    parser.add_argument("--s2v_ref_source_id", type=float, default=1.0, help="Reference-frame source_id when S2V source_id RoPE is enabled.")
    parser.add_argument("--s2v_ref_rope_theta", type=float, default=10000.0, help="RoPE theta used for the S2V reference-frame source_id phase.")
    parser.add_argument("--s2v_ref_time_base", type=int, default=30, help="Minimum legacy time offset for the S2V reference frame.")
    parser.add_argument("--s2v_ref_time_margin", type=int, default=9, help="Legacy time offset margin after the target latent length for the S2V reference frame.")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            fix_frame_rate=args.fix_frame_rate,
            time_division_factor=4 if not args.framewise_decoding else 1,
            time_division_remainder=1 if not args.framewise_decoding else 0,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            "wantodance_music_path": ToAbsolutePath(args.dataset_base_path),
        }
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        tiled=args.tiled,
        tile_size=args.tile_size,
        tile_stride=args.tile_stride,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        dpo_beta=args.dpo_beta,
        dpo_lambda_sft=args.dpo_lambda_sft,
        dpo_reference_free=args.dpo_reference_free,
        dpo_ref_loss_key_chosen=args.dpo_ref_loss_key_chosen,
        dpo_ref_loss_key_rejected=args.dpo_ref_loss_key_rejected,
        s2v_ref_rope_mode=args.s2v_ref_rope_mode,
        s2v_ref_source_id=args.s2v_ref_source_id,
        s2v_ref_rope_theta=args.s2v_ref_rope_theta,
        s2v_ref_time_base=args.s2v_ref_time_base,
        s2v_ref_time_margin=args.s2v_ref_time_margin,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "dpo:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
        "dpo": launch_training_task,
        "dpo:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
