from .operators import *
import hashlib, json, logging, os, re
import torch, pandas


logger = logging.getLogger(__name__)


class UnifiedDataset(torch.utils.data.Dataset):
    cache_key_field = "_data_cache_key"

    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        post_processor=None,
        max_data_items=None,
        cache_manifest_required=False,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.post_processor = post_processor
        self.max_data_items = max_data_items
        self.cache_manifest_required = cache_manifest_required
        self.cache_manifest = None
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)
    
    @staticmethod
    def is_missing_file_value(value):
        if value is None:
            return True
        if isinstance(value, str):
            return value == ""
        if isinstance(value, (list, tuple, dict)):
            return False
        try:
            return bool(pandas.isna(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
        frame_rate=24, fix_frame_rate=False,
        frame_count_stride=None, frame_count_remainder=None,
        frame_count_rounding="floor", min_num_frames=1,
        max_frame_padding=None, log_first_n=0,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp", "bmp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                    frame_rate=frame_rate, fix_frame_rate=fix_frame_rate,
                    frame_count_stride=frame_count_stride,
                    frame_count_remainder=frame_count_remainder,
                    frame_count_rounding=frame_count_rounding,
                    min_num_frames=min_num_frames,
                    max_frame_padding=max_frame_padding,
                    log_first_n=log_first_n,
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        for file_name in sorted(os.listdir(path)):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            manifest_path = os.path.join(self.base_path, "_cache_manifest.json")
            if os.path.isfile(manifest_path):
                with open(manifest_path, "r") as f:
                    self.cache_manifest = json.load(f)
                if self.cache_manifest.get("status") != "complete":
                    raise RuntimeError(
                        f"Feature cache is not complete: {manifest_path} has "
                        f"status={self.cache_manifest.get('status')!r}."
                    )
            elif self.cache_manifest_required:
                raise FileNotFoundError(
                    f"Required feature-cache manifest does not exist: {manifest_path}."
                )
            else:
                logger.warning("Feature cache has no manifest: %s", manifest_path)
            self.search_for_cached_data_files(self.base_path)
            self.cached_data = sorted(self.cached_data)
            if not self.cached_data:
                raise RuntimeError(f"No .pth feature-cache files found under {self.base_path}.")
            if self.cache_manifest is not None:
                expected_files = self.cache_manifest.get("cached_files")
                if expected_files is not None and expected_files != len(self.cached_data):
                    raise RuntimeError(
                        f"Feature-cache file count mismatch: manifest={expected_files}, "
                        f"found={len(self.cached_data)} under {self.base_path}."
                    )
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    @classmethod
    def normalize_cache_key_value(cls, value):
        if cls.is_missing_file_value(value):
            return None
        if isinstance(value, dict):
            return {str(key): cls.normalize_cache_key_value(value[key]) for key in sorted(value)}
        if isinstance(value, (list, tuple)):
            return [cls.normalize_cache_key_value(item) for item in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except ValueError:
                pass
        return value

    @classmethod
    def flatten_file_values(cls, value):
        if cls.is_missing_file_value(value):
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            values = []
            for key in sorted(value):
                values.extend(cls.flatten_file_values(value[key]))
            return values
        if isinstance(value, (list, tuple)):
            values = []
            for item in value:
                values.extend(cls.flatten_file_values(item))
            return values
        return []

    @staticmethod
    def safe_cache_key_component(value):
        value = str(value).replace("\\", "__").replace("/", "__")
        value = re.sub(r"[^A-Za-z0-9._=-]+", "_", value).strip("._-")
        return value[:96] if value else "empty"

    def build_data_cache_key(self, raw_data, raw_data_id):
        file_components = []
        for key in self.data_file_keys:
            if key not in raw_data:
                continue
            for value in self.flatten_file_values(raw_data[key]):
                file_components.append(f"{key}={value}")

        readable = "__".join(self.safe_cache_key_component(value) for value in file_components)
        if not readable:
            readable = f"item_{raw_data_id:08d}"
        readable = readable[:160].strip("._-") or f"item_{raw_data_id:08d}"

        payload = {
            "index": raw_data_id,
            "data_file_keys": list(self.data_file_keys),
            "metadata": self.normalize_cache_key_value(raw_data),
        }
        digest = hashlib.sha1(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()[:16]
        return f"{raw_data_id:08d}__{readable}__{digest}"

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            raw_data_id = data_id % len(self.data)
            data = self.data[raw_data_id].copy()
            data[self.cache_key_field] = self.build_data_cache_key(data, raw_data_id)
            for key in self.data_file_keys:
                if key in data:
                    if self.is_missing_file_value(data[key]):
                        data[key] = None
                    elif key in self.special_operator_map:
                        data[key] = self.special_operator_map[key](data[key])
                    elif key in self.data_file_keys:
                        data[key] = self.main_data_operator(data[key])
            if self.post_processor is not None:
                data = self.post_processor(data)
        return data

    def __len__(self):
        if self.max_data_items is not None:
            return self.max_data_items
        elif self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
