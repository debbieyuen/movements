import json
from pathlib import Path

import numpy as np
import pytest

from server.protocol import (
    COORD_ZUP_XFWD,
    DEFAULT_DEPTH_SCALE,
    build_live_frame,
    mp_world_to_zup,
)

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "protocol" / "pose_protocol.schema.json"


def _rotation_matrix(depth_scale: float = 1.0) -> np.ndarray:
    """Recover the linear map from the implementation itself."""
    basis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    cols = [mp_world_to_zup([[*e, 1.0]], depth_scale)[0][:3] for e in basis]
    return np.array(cols).T


def test_transform_is_proper_rotation():
    R = _rotation_matrix(depth_scale=1.0)
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)


def test_head_above_hips_maps_to_positive_z():
    # MediaPipe y is image-DOWN: the nose sits at negative y relative to hips.
    nose_mp = [0.0, -0.6, 0.0, 0.9]
    (out,) = mp_world_to_zup([nose_mp])
    assert out[2] > 0  # Z up


def test_leaning_toward_camera_maps_to_positive_x():
    # Moving toward the camera decreases MediaPipe z.
    near = mp_world_to_zup([[0.0, 0.0, -0.5, 1.0]])[0]
    far = mp_world_to_zup([[0.0, 0.0, 0.5, 1.0]])[0]
    assert near[0] > far[0]


def test_subject_left_maps_to_positive_y():
    # Subject's left shoulder appears image-right (positive x_mp).
    (out,) = mp_world_to_zup([[0.3, 0.0, 0.0, 1.0]])
    assert out[1] > 0


def test_depth_scale_applied_and_recorded():
    (out,) = mp_world_to_zup([[0.0, 0.0, 1.0, 1.0]], depth_scale=0.35)
    assert np.isclose(out[0], -0.35)


def test_dict_and_array_landmarks_agree():
    as_dict = mp_world_to_zup([{"x": 0.1, "y": -0.2, "z": 0.3, "visibility": 0.7}])
    as_array = mp_world_to_zup([[0.1, -0.2, 0.3, 0.7]])
    assert as_dict == as_array


def test_missing_visibility_defaults_to_one():
    (out,) = mp_world_to_zup([{"x": 0.0, "y": 0.0, "z": 0.0}])
    assert out[3] == 1.0


def _legacy_frame():
    return {
        "sessionId": "abc123",
        "role": "front-camera",
        "frameIndex": 42,
        "timeMs": 1234.5,
        "unixMs": 1753980041233,
        "worldLandmarks": [{"x": 0.0, "y": -0.1, "z": 0.05, "visibility": 0.9}] * 33,
    }


def test_build_live_frame_shape():
    frame = build_live_frame(_legacy_frame(), server_unix_ms=1753980041301)
    assert frame is not None
    assert frame["coord"] == COORD_ZUP_XFWD
    assert frame["seq"] == 42
    assert frame["tMs"] == 1234.5
    assert frame["meta"]["depthScale"] == DEFAULT_DEPTH_SCALE
    assert len(frame["world"]) == 33
    assert len(frame["world"][0]) == 4


def test_build_live_frame_rejects_missing_world():
    bad = _legacy_frame()
    del bad["worldLandmarks"]
    assert build_live_frame(bad, server_unix_ms=0) is None


def test_live_frame_matches_json_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())
    frame = build_live_frame(_legacy_frame(), server_unix_ms=1753980041301)
    resolver_schema = {**schema, "$ref": "#/$defs/poseFrameLive"}
    jsonschema.validate(frame, resolver_schema)
