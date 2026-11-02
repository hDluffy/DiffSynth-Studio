"""Offline single-identity annotations attached to the existing feature cache."""
import numpy as np
import torch
from .alignment import estimate_alignment, align_faces
from .arcface import load_arcface, identity_embedding, file_sha256

CACHE_VERSION = 1


class FaceIDAnnotator:
    def __init__(self, config, device):
        from insightface.model_zoo import get_model
        import onnxruntime as ort
        self.config = config
        self.device = device
        # Local explicit file; never silently download a different detector.
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        self.detector = get_model(config.faceid_detector_path, providers=['CPUExecutionProvider'],
                                  sess_options=session_options)
        if self.detector is None or getattr(self.detector, 'taskname', None) != 'detection':
            raise ValueError('faceid_detector_path must be a SCRFD detection ONNX, not a recognition model')
        self.detector.prepare(ctx_id=-1, input_size=(640, 640), det_thresh=config.faceid_det_threshold)
        self.encoder = load_arcface(config.faceid_weights, device)
        self.fingerprint = file_sha256(config.faceid_weights)
        self.detector_fingerprint = file_sha256(config.faceid_detector_path)

    def face(self, image):
        rgb = np.asarray(image.convert('RGB')).copy()
        boxes, points = self.detector.detect(rgb[:, :, ::-1].copy(), max_num=0)
        # The dataset promises a single face: ambiguity is masked, not guessed.
        if points is None or len(boxes) != 1:
            return None
        x0, y0, x1, y1, confidence = boxes[0]
        if min(x1 - x0, y1 - y0) < self.config.faceid_min_face_size:
            return None
        try:
            matrix = estimate_alignment(points[0])
        except (ValueError, np.linalg.LinAlgError):
            return None
        if not torch.isfinite(matrix).all():
            return None
        pixels = torch.from_numpy(rgb).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).to(self.device)
        crop = align_faces(pixels, matrix.unsqueeze(0).to(self.device))
        with torch.no_grad():
            embedding = identity_embedding(self.encoder, crop)[0].cpu()
        return matrix, embedding, float(confidence)

    def __call__(self, frames, reference):
        if len(frames) != 81:
            raise ValueError(f'FaceID v1 requires exactly 81 frames, got {len(frames)}')
        ref = self.face(reference)
        matrices = torch.zeros(81, 2, 3)
        valid = torch.zeros(81, dtype=torch.bool)
        quality = torch.zeros(81)
        embedding = torch.zeros(512) if ref is None else ref[1]
        if ref is not None:
            # Two candidates per 4-frame decoder chunk; supervise the whole timeline.
            for index in range(1, 81):
                if index % 4 not in (2, 0):
                    continue
                face = self.face(frames[index])
                if face is None:
                    continue
                matrix, current_embedding, confidence = face
                if (embedding @ current_embedding).item() < self.config.faceid_min_similarity:
                    continue
                matrices[index] = matrix
                valid[index] = True
                quality[index] = confidence
        return dict(version=CACHE_VERSION, model_sha256=self.fingerprint,
                    detector_sha256=self.detector_fingerprint, reference_valid=ref is not None,
                    reference_embedding=embedding, inverse_matrices=matrices,
                    valid=valid, quality=quality, num_frames=81,
                    height=frames[0].height, width=frames[0].width)
