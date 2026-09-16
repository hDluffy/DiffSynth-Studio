import math, warnings
from dataclasses import dataclass

import numpy as np
import torch, torchvision, imageio, os
import imageio.v3 as iio
from PIL import Image

class VideoFrames(list):
    """A frame list carrying the sampling decision used to create it."""

    def __init__(self, frames=(), sampling_info=None):
        super().__init__(frames)
        self.sampling_info = {} if sampling_info is None else sampling_info


@dataclass(frozen=True)
class FrameSamplingPlan:
    original_num_frames: int
    original_frame_rate: float
    available_num_frames: int
    selected_num_frames: int
    padded_num_frames: int
    truncated_num_frames: int
    target_frame_rate: float
    rounding: str

    @property
    def target_duration_seconds(self):
        return (self.selected_num_frames - 1) / self.target_frame_rate

    def as_dict(self):
        return {
            "original_num_frames": self.original_num_frames,
            "original_frame_rate": self.original_frame_rate,
            "available_num_frames": self.available_num_frames,
            "selected_num_frames": self.selected_num_frames,
            "padded_num_frames": self.padded_num_frames,
            "truncated_num_frames": self.truncated_num_frames,
            "target_frame_rate": self.target_frame_rate,
            "target_duration_seconds": self.target_duration_seconds,
            "rounding": self.rounding,
        }


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data


class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)


class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)


class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)


class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True, convert_RGBA=False):
        self.convert_RGB = convert_RGB
        self.convert_RGBA = convert_RGBA
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        if self.convert_RGBA: image = image.convert("RGBA")
        return image


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    

class FrameSamplerByRateMixin:
    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_rate=24,
        fix_frame_rate=False,
        frame_count_stride=None,
        frame_count_remainder=None,
        frame_count_rounding="floor",
        min_num_frames=1,
        max_frame_padding=None,
        log_first_n=0,
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_rate = frame_rate
        self.fix_frame_rate = fix_frame_rate
        self.frame_count_stride = time_division_factor if frame_count_stride is None else frame_count_stride
        self.frame_count_remainder = time_division_remainder if frame_count_remainder is None else frame_count_remainder
        self.frame_count_rounding = frame_count_rounding
        self.min_num_frames = min_num_frames
        self.max_frame_padding = max_frame_padding
        self.log_first_n = log_first_n
        self._num_sampling_logs = 0
        self._validate_frame_sampling_config()

    def _validate_frame_sampling_config(self):
        if self.num_frames < 1:
            raise ValueError(f"num_frames must be positive, got {self.num_frames}.")
        if self.frame_rate <= 0:
            raise ValueError(f"frame_rate must be positive, got {self.frame_rate}.")
        if self.frame_count_stride < 1:
            raise ValueError(f"frame_count_stride must be positive, got {self.frame_count_stride}.")
        if not 0 <= self.frame_count_remainder < self.frame_count_stride:
            raise ValueError(
                "frame_count_remainder must be in [0, frame_count_stride), got "
                f"{self.frame_count_remainder} and {self.frame_count_stride}."
            )
        if self.frame_count_rounding not in {"floor", "nearest", "ceil"}:
            raise ValueError(
                "frame_count_rounding must be one of floor, nearest, ceil, got "
                f"{self.frame_count_rounding}."
            )
        if self.min_num_frames < 1 or self.min_num_frames > self.num_frames:
            raise ValueError(
                f"min_num_frames must be in [1, num_frames], got {self.min_num_frames} and {self.num_frames}."
            )
        if self.max_frame_padding is not None and self.max_frame_padding < 0:
            raise ValueError(f"max_frame_padding cannot be negative, got {self.max_frame_padding}.")

    def get_reader(self, data: str):
        return imageio.get_reader(data)

    def get_available_num_frames(self, reader):
        total_original_frames = int(reader.count_frames())
        if total_original_frames < 1:
            raise ValueError("The video has no decodable frames.")
        if not self.fix_frame_rate:
            return total_original_frames
        meta_data = reader.get_meta_data()
        raw_frame_rate = float(meta_data.get("fps", 0))
        if not math.isfinite(raw_frame_rate) or raw_frame_rate <= 0:
            raise ValueError(f"Invalid video frame rate: {raw_frame_rate}.")
        # Sample timestamps start at zero. Container duration often includes
        # one extra frame interval, so derive availability from the timestamp
        # of the final decodable frame.
        last_frame_timestamp = (total_original_frames - 1) / raw_frame_rate
        return int(math.floor(last_frame_timestamp * self.frame_rate + 1e-8) + 1)

    def _ceil_to_valid_count(self, value):
        stride, remainder = self.frame_count_stride, self.frame_count_remainder
        return int(math.ceil((value - remainder) / stride) * stride + remainder)

    def _floor_to_valid_count(self, value):
        stride, remainder = self.frame_count_stride, self.frame_count_remainder
        return int(math.floor((value - remainder) / stride) * stride + remainder)

    def select_num_frames(self, available_num_frames):
        min_valid = self._ceil_to_valid_count(self.min_num_frames)
        max_valid = self._floor_to_valid_count(self.num_frames)
        if min_valid > max_valid:
            raise ValueError(
                "No valid frame count exists for the configured range: "
                f"min_num_frames={self.min_num_frames}, num_frames={self.num_frames}, "
                f"constraint={self.frame_count_stride}n+{self.frame_count_remainder}."
            )

        floor_count = min(max(self._floor_to_valid_count(available_num_frames), min_valid), max_valid)
        ceil_count = min(max(self._ceil_to_valid_count(available_num_frames), min_valid), max_valid)
        if self.frame_count_rounding == "floor":
            selected_num_frames = floor_count
        elif self.frame_count_rounding == "ceil":
            selected_num_frames = ceil_count
        else:
            floor_distance = abs(available_num_frames - floor_count)
            ceil_distance = abs(ceil_count - available_num_frames)
            # Prefer truncation on an exact tie to avoid fabricating frames.
            selected_num_frames = floor_count if floor_distance <= ceil_distance else ceil_count

        padded_num_frames = max(0, selected_num_frames - available_num_frames)
        if self.max_frame_padding is not None and padded_num_frames > self.max_frame_padding:
            raise ValueError(
                f"Selecting {selected_num_frames} frames requires padding {padded_num_frames} frames, "
                f"which exceeds max_frame_padding={self.max_frame_padding}. "
                "Use floor rounding, increase the limit, or filter this sample."
            )
        return selected_num_frames

    def get_sampling_plan(self, reader):
        meta_data = reader.get_meta_data()
        total_original_frames = int(reader.count_frames())
        raw_frame_rate = float(meta_data.get("fps", self.frame_rate))
        available_num_frames = self.get_available_num_frames(reader)
        selected_num_frames = self.select_num_frames(available_num_frames)
        return FrameSamplingPlan(
            original_num_frames=total_original_frames,
            original_frame_rate=raw_frame_rate,
            available_num_frames=available_num_frames,
            selected_num_frames=selected_num_frames,
            padded_num_frames=max(0, selected_num_frames - available_num_frames),
            truncated_num_frames=max(0, available_num_frames - selected_num_frames),
            target_frame_rate=float(self.frame_rate if self.fix_frame_rate else raw_frame_rate),
            rounding=self.frame_count_rounding,
        )

    def get_num_frames(self, reader):
        return self.get_sampling_plan(reader).selected_num_frames

    def map_single_frame_id(self, new_sequence_id: int, raw_frame_rate: float, total_raw_frames: int) -> int:
        if not self.fix_frame_rate:
            return min(new_sequence_id, total_raw_frames - 1)
        target_time_in_seconds = new_sequence_id / self.frame_rate
        raw_frame_index_float = target_time_in_seconds * raw_frame_rate
        frame_id = int(round(raw_frame_index_float))
        return min(frame_id, total_raw_frames - 1)


class LoadVideo(DataProcessingOperator, FrameSamplerByRateMixin):
    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_processor=lambda x: x,
        frame_rate=24,
        fix_frame_rate=False,
        frame_count_stride=None,
        frame_count_remainder=None,
        frame_count_rounding="floor",
        min_num_frames=1,
        max_frame_padding=None,
        log_first_n=0,
    ):
        FrameSamplerByRateMixin.__init__(
            self,
            num_frames,
            time_division_factor,
            time_division_remainder,
            frame_rate,
            fix_frame_rate,
            frame_count_stride,
            frame_count_remainder,
            frame_count_rounding,
            min_num_frames,
            max_frame_padding,
            log_first_n,
        )
        self.frame_processor = frame_processor

    def __call__(self, data: str):
        reader = self.get_reader(data)
        try:
            plan = self.get_sampling_plan(reader)
            frames = []
            for frame_id in range(plan.selected_num_frames):
                source_frame_id = self.map_single_frame_id(
                    frame_id, plan.original_frame_rate, plan.original_num_frames
                )
                frame = reader.get_data(source_frame_id)
                frame = Image.fromarray(frame)
                frame = self.frame_processor(frame)
                frames.append(frame)
        finally:
            reader.close()

        sampling_info = plan.as_dict()
        sampling_info["path"] = data
        if self._num_sampling_logs < self.log_first_n:
            print(
                "[DiffSynth data] Video sampling: "
                f"path={data} raw={plan.original_num_frames}@{plan.original_frame_rate:.6g}fps "
                f"available={plan.available_num_frames} selected={plan.selected_num_frames} "
                f"constraint={self.frame_count_stride}n+{self.frame_count_remainder} "
                f"rounding={plan.rounding} padded={plan.padded_num_frames} "
                f"truncated={plan.truncated_num_frames} duration={plan.target_duration_seconds:.6f}s",
                flush=True,
            )
            self._num_sampling_logs += 1
        return VideoFrames(frames, sampling_info=sampling_info)


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]


class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor

    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")


class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)


class LoadAudio(DataProcessingOperator):
    def __init__(self, sr=16000):
        self.sr = sr
        import librosa
        self.audio_loader = librosa.load
    
    def __call__(self, data: str):
        input_audio, sample_rate = self.audio_loader(data, sr=self.sr)
        return input_audio


class AlignAudioToVideo(DataProcessingOperator):
    """Align an audio waveform exactly to the sampled video time span.

    Video duration is defined by frame timestamps, i.e. ``(frames - 1) / fps``.
    The operator always returns exactly ``round(duration * sample_rate)`` audio
    samples. ``strict`` limits any automatic correction to the configured
    tolerance; ``trim_pad`` permits normal prefix trimming and bounded padding.
    """

    def __init__(
        self,
        video_key="video",
        audio_key="input_audio",
        frame_rate=16,
        sample_rate=16000,
        policy="trim_pad",
        tolerance_seconds=0.05,
        max_padding_seconds=0.5,
        max_trimming_seconds=None,
        log_first_n=8,
    ):
        if frame_rate <= 0 or sample_rate <= 0:
            raise ValueError("frame_rate and sample_rate must be positive.")
        if policy not in {"strict", "trim_pad"}:
            raise ValueError(f"Unsupported audio duration policy: {policy}.")
        if tolerance_seconds < 0:
            raise ValueError("tolerance_seconds cannot be negative.")
        if max_padding_seconds is not None and max_padding_seconds < 0:
            raise ValueError("max_padding_seconds cannot be negative.")
        if max_trimming_seconds is not None and max_trimming_seconds < 0:
            raise ValueError("max_trimming_seconds cannot be negative.")
        self.video_key = video_key
        self.audio_key = audio_key
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.policy = policy
        self.tolerance_seconds = tolerance_seconds
        self.max_padding_seconds = max_padding_seconds
        self.max_trimming_seconds = max_trimming_seconds
        self.log_first_n = log_first_n
        self._num_logs = 0

    @staticmethod
    def _pad_audio(audio, padding):
        if torch.is_tensor(audio):
            return torch.nn.functional.pad(audio, (0, padding))
        if isinstance(audio, np.ndarray):
            pad_width = [(0, 0)] * audio.ndim
            pad_width[-1] = (0, padding)
            return np.pad(audio, pad_width, mode="constant")
        raise TypeError(f"Unsupported audio type: {type(audio).__name__}.")

    def __call__(self, data):
        video = data.get(self.video_key)
        audio = data.get(self.audio_key)
        sample_id = data.get("_data_cache_key", "unknown")
        if video is None:
            raise ValueError(f"Cannot align audio without `{self.video_key}` for sample {sample_id}.")
        if audio is None:
            raise ValueError(f"Missing `{self.audio_key}` for sample {sample_id}.")
        if not hasattr(audio, "shape") or len(audio.shape) < 1:
            raise TypeError(f"Invalid audio object for sample {sample_id}: {type(audio).__name__}.")

        num_frames = len(video)
        if num_frames < 1:
            raise ValueError(f"Video contains no frames for sample {sample_id}.")
        sampling_info = getattr(video, "sampling_info", {})
        video_frame_rate = float(sampling_info.get("target_frame_rate", self.frame_rate))
        if not math.isfinite(video_frame_rate) or video_frame_rate <= 0:
            raise ValueError(f"Invalid sampled video frame rate for sample {sample_id}: {video_frame_rate}.")
        target_duration = (num_frames - 1) / video_frame_rate
        target_samples = int(round(target_duration * self.sample_rate))
        current_samples = int(audio.shape[-1])
        delta_samples = target_samples - current_samples
        delta_seconds = abs(delta_samples) / self.sample_rate

        if self.policy == "strict" and delta_seconds > self.tolerance_seconds:
            raise ValueError(
                f"Audio/video duration mismatch for sample {sample_id}: video={target_duration:.6f}s "
                f"({num_frames} frames at {video_frame_rate}fps), audio={current_samples / self.sample_rate:.6f}s. "
                f"The difference {delta_seconds:.6f}s exceeds tolerance={self.tolerance_seconds:.6f}s."
            )
        if delta_samples > 0:
            if self.max_padding_seconds is not None and delta_seconds > self.max_padding_seconds:
                raise ValueError(
                    f"Audio for sample {sample_id} needs {delta_seconds:.6f}s of zero padding, "
                    f"exceeding max_audio_padding_seconds={self.max_padding_seconds}."
                )
            audio = self._pad_audio(audio, delta_samples)
            action = "pad"
        elif delta_samples < 0:
            if self.max_trimming_seconds is not None and delta_seconds > self.max_trimming_seconds:
                raise ValueError(
                    f"Audio for sample {sample_id} needs {delta_seconds:.6f}s of trimming, "
                    f"exceeding max_audio_trimming_seconds={self.max_trimming_seconds}."
                )
            audio = audio[..., :target_samples]
            action = "trim"
        else:
            action = "none"

        data[self.audio_key] = audio
        data["sample_num_frames"] = num_frames
        data["sample_duration_seconds"] = target_duration
        data["audio_num_samples"] = target_samples
        data["audio_sample_rate"] = self.sample_rate
        data["video_frame_rate"] = video_frame_rate
        data["video_sampling_info"] = sampling_info
        if self._num_logs < self.log_first_n:
            sampling_info = data["video_sampling_info"]
            print(
                "[DiffSynth data] Audio/video alignment: "
                f"sample={sample_id} frames={num_frames} fps={video_frame_rate:.6g} "
                f"duration={target_duration:.6f}s audio_before={current_samples} "
                f"audio_after={target_samples} action={action} "
                f"video_padded={sampling_info.get('padded_num_frames', 0)} "
                f"video_truncated={sampling_info.get('truncated_num_frames', 0)}",
                flush=True,
            )
            self._num_logs += 1
        return data


class LoadAudioWithTorchaudio(DataProcessingOperator, FrameSamplerByRateMixin):

    def __init__(self, num_frames=121, time_division_factor=8, time_division_remainder=1, frame_rate=24, fix_frame_rate=True):
        FrameSamplerByRateMixin.__init__(self, num_frames, time_division_factor, time_division_remainder, frame_rate, fix_frame_rate)
        import torchaudio
        self.audio_loader = torchaudio.load

    def __call__(self, data: str):
        try:
            reader = self.get_reader(data)
            num_frames = self.get_num_frames(reader)
            duration = num_frames / self.frame_rate
            waveform, sample_rate = self.audio_loader(data)
            target_samples = int(duration * sample_rate)
            current_samples = waveform.shape[-1]
            if current_samples > target_samples:
                waveform = waveform[..., :target_samples]
            elif current_samples < target_samples:
                padding = target_samples - current_samples
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            return waveform, sample_rate
        except:
            warnings.warn(f"Cannot load audio in {data}. The audio will be `None`.")
            return None


class LoadPureAudioWithTorchaudio(DataProcessingOperator):

    def __init__(self, target_sample_rate=None, target_duration=None):
        self.target_sample_rate = target_sample_rate
        self.target_duration = target_duration
        self.resample = True if target_sample_rate is not None else False
        from diffsynth.utils.data.audio import read_audio
        self.audio_loader = read_audio

    def __call__(self, data: str):
        try:
            waveform, sample_rate = self.audio_loader(data, resample=self.resample, resample_rate=self.target_sample_rate)
            if self.target_duration is not None:
                target_samples = int(self.target_duration * sample_rate)
                current_samples = waveform.shape[-1]
                if current_samples > target_samples:
                    waveform = waveform[..., :target_samples]
                elif current_samples < target_samples:
                    padding = target_samples - current_samples
                    waveform = torch.nn.functional.pad(waveform, (0, padding))
            return waveform, sample_rate
        except Exception as e:
            print(f"Cannot load audio in {data} due to {e}. The audio will be `None`.")
            return None
