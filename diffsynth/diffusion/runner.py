import hashlib, importlib, json, os, socket
from collections import Counter

import torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from diffsynth.core import OffloadTrainingManager


_CACHE_MANIFEST_NAME = "_cache_manifest.json"
_CACHE_CONFIG_KEYS = (
    "dataset_base_path", "dataset_metadata_path", "data_file_keys",
    "height", "width", "max_pixels", "num_frames", "frame_rate",
    "fix_frame_rate", "frame_count_stride", "frame_count_remainder",
    "frame_count_rounding", "min_num_frames", "max_frame_padding",
    "audio_sample_rate", "audio_duration_policy",
    "audio_duration_tolerance_seconds", "max_audio_padding_seconds",
    "max_audio_trimming_seconds", "model_paths",
    "model_id_with_origin_paths", "audio_processor_path", "extra_inputs",
    "tiled", "tile_size", "tile_stride", "fp8_models", "offload_models",
    "task",
)


def _process_memory_mib():
    values = {}
    try:
        with open("/proc/self/status", "r") as status_file:
            for line in status_file:
                key, _, value = line.partition(":")
                if key in {"VmRSS", "VmHWM"}:
                    values[key] = int(value.strip().split()[0]) / 1024
    except (FileNotFoundError, OSError, ValueError):
        pass
    return values


def _log_memory_snapshot(accelerator, phase):
    process_memory = _process_memory_mib()
    cuda_allocated = 0.0
    cuda_reserved = 0.0
    if torch.cuda.is_available() and accelerator.device.type == "cuda":
        cuda_allocated = torch.cuda.memory_allocated(accelerator.device) / (1024 ** 2)
        cuda_reserved = torch.cuda.memory_reserved(accelerator.device) / (1024 ** 2)
    print(
        "Memory snapshot: "
        f"phase={phase}, host={socket.gethostname()}, "
        f"rank={accelerator.process_index}, local_rank={accelerator.local_process_index}, "
        f"cpu_rss_mib={process_memory.get('VmRSS', -1):.1f}, "
        f"cpu_hwm_mib={process_memory.get('VmHWM', -1):.1f}, "
        f"cuda_allocated_mib={cuda_allocated:.1f}, "
        f"cuda_reserved_mib={cuda_reserved:.1f}",
        flush=True,
    )


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _sha256_file(path):
    if path is None or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_config(args, dataset):
    values = vars(args) if args is not None else {}
    config = {key: _json_safe(values.get(key)) for key in _CACHE_CONFIG_KEYS}
    config["metadata_sha256"] = _sha256_file(getattr(dataset, "metadata_path", None))
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return config, hashlib.sha256(payload).hexdigest()


def _atomic_write_json(path, payload):
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _count_cache_files(path):
    return sum(
        file_name.endswith(".pth")
        for _, _, file_names in os.walk(path)
        for file_name in file_names
    )


def _move_cache_payload_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_move_cache_payload_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_move_cache_payload_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _move_cache_payload_to_cpu(item) for key, item in value.items()}
    return value


def _payload_keys(value):
    if isinstance(value, dict):
        return sorted(value)
    if isinstance(value, (list, tuple)):
        return [_payload_keys(item) for item in value]
    return type(value).__name__


def get_optimizer_class(customized_optimizer=None):
    if customized_optimizer is None:
        return torch.optim.AdamW
    else:
        module_name, class_name = customized_optimizer.rsplit(".", 1)
        module = importlib.import_module(module_name)
        print(f"Customized opimizer `{customized_optimizer}` imported.")
        return getattr(module, class_name)


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    enable_model_cpu_offload: bool = False,
    enable_optimizer_cpu_offload: bool = False,
    cpu_offload_split_threshold: int = None,
    customized_optimizer: str = None,
    args = None,
    **kwargs,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        customized_optimizer = args.customized_optimizer

    _log_memory_snapshot(accelerator, "training_entry")
    optimizer_class = get_optimizer_class(customized_optimizer)
    optimizer = optimizer_class(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    _log_memory_snapshot(accelerator, "optimizer_and_dataloader_created")

    if enable_model_cpu_offload:
        _log_memory_snapshot(accelerator, "before_accelerator_prepare")
        optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
        _log_memory_snapshot(accelerator, "after_accelerator_prepare")
        model.pipe.device = accelerator.device
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
    else:
        _log_memory_snapshot(accelerator, "before_model_to_device")
        model.to(device=accelerator.device)
        _log_memory_snapshot(accelerator, "after_model_to_device")
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
        _log_memory_snapshot(accelerator, "after_accelerator_prepare")

    initialize_deepspeed_gradient_checkpointing(accelerator)
    first_step = True
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            if first_step:
                _log_memory_snapshot(accelerator, "first_step_data_loaded")
            with accelerator.accumulate(model):
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                if first_step:
                    _log_memory_snapshot(accelerator, "first_step_after_forward")
                accelerator.backward(loss)
                if first_step:
                    _log_memory_snapshot(accelerator, "first_step_after_backward")
                if enable_model_cpu_offload:
                    offload_manager.after_backward()
                optimizer.step()
                if first_step:
                    _log_memory_snapshot(accelerator, "first_step_after_optimizer_step")
                scheduler.step()
                optimizer.zero_grad()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
            first_step = False
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
    **kwargs,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        resume_feature_cache = args.resume_feature_cache
    else:
        resume_feature_cache = False

    output_path = model_logger.output_path
    os.makedirs(output_path, exist_ok=True)
    if getattr(dataset, "repeat", 1) != 1:
        raise ValueError(
            "Feature-cache generation requires dataset_repeat=1 to avoid duplicate cache keys."
        )
    source_items = len(dataset)
    manifest_path = os.path.join(output_path, _CACHE_MANIFEST_NAME)
    config, config_fingerprint = _cache_config(args, dataset)
    existing_manifest = None
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r") as f:
            existing_manifest = json.load(f)
    existing_cache_files = _count_cache_files(output_path)
    if existing_cache_files and not resume_feature_cache:
        raise RuntimeError(
            f"Feature-cache output already contains {existing_cache_files} .pth files: {output_path}. "
            "Use a fresh output path or pass --resume_feature_cache."
        )
    if resume_feature_cache and existing_cache_files:
        if existing_manifest is None:
            raise RuntimeError(
                f"Cannot safely resume cache without {_CACHE_MANIFEST_NAME}: {output_path}."
            )
        if existing_manifest.get("config_fingerprint") != config_fingerprint:
            raise RuntimeError(
                "Feature-cache configuration mismatch. Use a new output path instead of mixing caches. "
                f"existing={existing_manifest.get('config_fingerprint')}, current={config_fingerprint}."
            )
        if existing_manifest.get("world_size") != accelerator.num_processes:
            raise RuntimeError(
                "Feature-cache world size mismatch. Resume with the same number of processes or use "
                "a new output path. "
                f"existing={existing_manifest.get('world_size')}, "
                f"current={accelerator.num_processes}."
            )
        if (
            existing_manifest.get("status") == "complete"
            and existing_manifest.get("cached_files") == existing_cache_files
        ):
            accelerator.print(
                f"Feature cache is already complete with {existing_cache_files} files: {output_path}"
            )
            return

    if accelerator.is_main_process:
        _atomic_write_json(
            manifest_path,
            {
                "format_version": 1,
                "status": "building",
                "config_fingerprint": config_fingerprint,
                "config": config,
                "source_items": source_items,
                "world_size": accelerator.num_processes,
                "cached_files_before_run": existing_cache_files,
            },
        )
    accelerator.wait_for_everyone()

    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers
    )
    # Cache generation does not run backward and therefore does not need a DDP
    # model wrapper. Uneven sharding prevents Accelerate from duplicating tail
    # samples when the dataset size is not divisible by the process count.
    accelerator.even_batches = False
    if enable_model_cpu_offload:
        dataloader = accelerator.prepare(dataloader)
        offload_manager = OffloadTrainingManager(
            model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold
        )
        model.pipe.device = accelerator.device
    else:
        model.to(device=accelerator.device)
        dataloader = accelerator.prepare(dataloader)

    cache_key_field = getattr(dataset, "cache_key_field", "_data_cache_key")
    frame_count_distribution = Counter()
    saved_files = 0
    reused_files = 0
    logged_payload = False
    folder = os.path.join(output_path, str(accelerator.process_index))
    os.makedirs(folder, exist_ok=True)
    for data_id, data in enumerate(tqdm(dataloader)):
        cache_key = None
        if isinstance(data, dict):
            cache_key = data.pop(cache_key_field, None)
            sample_num_frames = data.get("sample_num_frames")
            if sample_num_frames is not None:
                frame_count_distribution[int(sample_num_frames)] += 1
        file_name = f"{cache_key}.pth" if cache_key else f"{data_id:08d}.pth"
        save_path = os.path.join(folder, file_name)
        if resume_feature_cache and os.path.isfile(save_path) and os.path.getsize(save_path) > 0:
            reused_files += 1
            continue
        tmp_save_path = f"{save_path}.rank{accelerator.process_index}.pid{os.getpid()}.tmp"
        with torch.no_grad():
            cache_payload = model(data)
            cache_payload = _move_cache_payload_to_cpu(cache_payload)
            if not logged_payload:
                print(
                    f"Feature-cache payload rank={accelerator.process_index}: "
                    f"keys={_payload_keys(cache_payload)}"
                )
                logged_payload = True
            try:
                torch.save(cache_payload, tmp_save_path)
                os.replace(tmp_save_path, save_path)
            finally:
                if os.path.exists(tmp_save_path):
                    os.remove(tmp_save_path)
            saved_files += 1
            if enable_model_cpu_offload:
                offload_manager.after_backward()

    rank_summary = {
        "rank": accelerator.process_index,
        "saved_files": saved_files,
        "reused_files": reused_files,
        "frame_count_distribution": {
            str(key): frame_count_distribution[key] for key in sorted(frame_count_distribution)
        },
    }
    _atomic_write_json(
        os.path.join(output_path, f"_cache_summary_rank_{accelerator.process_index}.json"),
        rank_summary,
    )
    print(f"Feature-cache rank summary: {rank_summary}")
    accelerator.wait_for_everyone()

    cached_files = _count_cache_files(output_path)
    if cached_files != source_items:
        raise RuntimeError(
            "Feature-cache file count does not match the source dataset after processing: "
            f"expected={source_items}, found={cached_files}. "
            "Check for duplicate cache keys or use a fresh output path."
        )

    if accelerator.is_main_process:
        summaries = []
        merged_distribution = Counter()
        for process_index in range(accelerator.num_processes):
            summary_path = os.path.join(output_path, f"_cache_summary_rank_{process_index}.json")
            with open(summary_path, "r") as f:
                summary = json.load(f)
            summaries.append(summary)
            merged_distribution.update(
                {int(key): value for key, value in summary["frame_count_distribution"].items()}
            )
        _atomic_write_json(
            manifest_path,
            {
                "format_version": 1,
                "status": "complete",
                "config_fingerprint": config_fingerprint,
                "config": config,
                "source_items": source_items,
                "world_size": accelerator.num_processes,
                "cached_files": cached_files,
                "frame_count_distribution": {
                    str(key): merged_distribution[key] for key in sorted(merged_distribution)
                },
                "rank_summaries": summaries,
            },
        )
        print(
            f"Feature cache completed: path={output_path}, files={cached_files}, "
            f"frame_count_distribution={dict(sorted(merged_distribution.items()))}"
        )
    accelerator.wait_for_everyone()

def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
