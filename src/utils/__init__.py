"""Utility modules for MotoGP audio syncing."""

from .audio_utils import (
    get_duration,
    get_audio_stream_count,
    extract_wav,
    extract_seg,
    load_fp_wav,
    concat_segments_to_mka,
)
from .sting_detection import find_sting, find_all_transitions
from .watermark_detection import (
    get_video_dimensions,
    build_watermark_template,
    find_break_end_via_watermark,
)

__all__ = [
    "get_duration",
    "get_audio_stream_count",
    "extract_wav",
    "extract_seg",
    "load_fp_wav",
    "concat_segments_to_mka",
    "find_sting",
    "find_all_transitions",
    "get_video_dimensions",
    "build_watermark_template",
    "find_break_end_via_watermark",
]
