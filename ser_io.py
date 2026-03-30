"""
speckle_suite.ser_io
====================
Low-level SER file parser.  No Qt dependency; only stdlib + numpy.

Public API
----------
SERHeader           dataclass with all header fields
parse_ser_header()  parse 178-byte header bytes → SERHeader
ser_frame_iter()    generator yielding float32 frames
COLOR_MONO / COLOR_BAYER_RGGB / COLOR_BGR / COLOR_RGB  colour-mode constants
"""

import struct
from dataclasses import dataclass

import numpy as np

# ── Colour-mode constants ──────────────────────────────────────────────────

COLOR_MONO       = 0
COLOR_BAYER_RGGB = 8
COLOR_BGR        = 100
COLOR_RGB        = 101

# ── SER header ─────────────────────────────────────────────────────────────

_HEADER_SIZE = 178   # bytes


@dataclass
class SERHeader:
    file_id:       str
    lu_id:         int
    color_id:      int
    little_endian: int
    image_width:   int
    image_height:  int
    pixel_depth:   int
    frame_count:   int
    observer:      str
    instrument:    str
    telescope:     str
    date_time:     int
    date_time_utc: int

    @property
    def bytes_per_pixel(self) -> int:
        return 2 if self.pixel_depth > 8 else 1

    @property
    def frame_size(self) -> int:
        return self.image_width * self.image_height * self.bytes_per_pixel

    @property
    def is_colour(self) -> bool:
        return self.color_id in (COLOR_BGR, COLOR_RGB)


def parse_ser_header(data: bytes) -> SERHeader:
    """Parse 178 header bytes and return a SERHeader."""
    if len(data) < _HEADER_SIZE:
        raise ValueError(f"SER header too short: {len(data)} < {_HEADER_SIZE}")
    return SERHeader(
        file_id       = data[0:14].decode('ascii', errors='ignore').rstrip('\x00'),
        lu_id         = struct.unpack_from('<i', data, 14)[0],
        color_id      = struct.unpack_from('<i', data, 18)[0],
        little_endian = struct.unpack_from('<i', data, 22)[0],
        image_width   = struct.unpack_from('<i', data, 26)[0],
        image_height  = struct.unpack_from('<i', data, 30)[0],
        pixel_depth   = struct.unpack_from('<i', data, 34)[0],
        frame_count   = struct.unpack_from('<i', data, 38)[0],
        observer      = data[42:82].decode('ascii',  errors='ignore').rstrip('\x00'),
        instrument    = data[82:122].decode('ascii', errors='ignore').rstrip('\x00'),
        telescope     = data[122:162].decode('ascii', errors='ignore').rstrip('\x00'),
        date_time     = struct.unpack_from('<q', data, 162)[0],
        date_time_utc = struct.unpack_from('<q', data, 170)[0],
    )


def ser_frame_iter(filepath: str, header: SERHeader):
    """
    Yield each frame from a SER file as a float32 ndarray (H × W).

    For colour (BGR / RGB) files the channels are averaged to produce
    a monochrome luminance frame, consistent with what the drift and
    preprocess backends expect.
    """
    dtype = np.uint16 if header.bytes_per_pixel == 2 else np.uint8
    fsize = header.frame_size

    with open(filepath, 'rb') as f:
        f.seek(_HEADER_SIZE)
        for _ in range(header.frame_count):
            raw = f.read(fsize)
            if len(raw) < fsize:
                break
            arr = np.frombuffer(raw, dtype=dtype).reshape(
                header.image_height, header.image_width
            )
            if header.is_colour:
                arr = arr.reshape(
                    header.image_height, header.image_width // 3, 3
                ).mean(axis=2)
            yield arr.astype(np.float32)


def read_ser_header(filepath: str) -> SERHeader:
    """Read and return the header from a SER file (no frames loaded)."""
    with open(filepath, 'rb') as f:
        return parse_ser_header(f.read(_HEADER_SIZE))


def read_ser_header_and_timestamps(
        filepath: str,
) -> tuple[SERHeader, np.ndarray | None]:
    """
    Read the SER header and the per-frame timestamp trailer.

    The trailer is 8 bytes × frame_count appended after all frame data.
    Timestamps are Windows FILETIME (100-ns ticks); we convert to elapsed
    seconds from the first frame.

    Returns
    -------
    header         : SERHeader
    timestamps_sec : float64 ndarray of elapsed seconds, or None if absent /
                     invalid (timestamps not recorded by capture software).
    """
    with open(filepath, 'rb') as f:
        header = parse_ser_header(f.read(_HEADER_SIZE))
        trailer_offset = _HEADER_SIZE + header.frame_count * header.frame_size
        timestamps_sec = None
        try:
            f.seek(trailer_offset)
            raw_ts = f.read(header.frame_count * 8)
            if len(raw_ts) == header.frame_count * 8:
                ts = np.frombuffer(raw_ts, dtype=np.int64).copy()
                if np.all(ts > 0) and np.all(np.diff(ts) > 0):
                    # Convert Windows FILETIME 100-ns ticks → elapsed seconds
                    timestamps_sec = (ts - ts[0]).astype(np.float64) * 1e-7
        except Exception:
            pass

    return header, timestamps_sec
