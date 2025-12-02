import cv2
import logging

logger = logging.getLogger(__name__)

def seek_and_get_ts(cap: cv2.VideoCapture, t_ms: float) -> tuple[bool, float | None]:
    """
    Seek to a timestamp in a video and return the actual timestamp after reading a frame.

    Parameters
    ----------
    cap : cv2.VideoCapture
        OpenCV video capture object.
    t_ms : float
        Timestamp to seek to in milliseconds.

    Returns
    -------
    ok : bool
        True if a frame was successfully read at or near the requested timestamp.
    actual_ts : float or None
        Actual timestamp (ms) of the decoded frame. None if read failed.

    Notes
    -----
    OpenCV's seeking may not land exactly on `t_ms` and may land on the nearest decodable frame.
    """
    if not cap.set(cv2.CAP_PROP_POS_MSEC, t_ms):
        logger.info(f"could not set capture device to {t_ms:.3f} milliseconds")
        return False, None

    ret, _ = cap.read()
    if not ret:
        logger.info(f"could not read capture device at {t_ms:.3f} milliseconds")
        return False, None

    actual_ts = cap.get(cv2.CAP_PROP_POS_MSEC)
    logger.info(f"actual timestamp {actual_ts:.3f} when setting capture device at {t_ms:.3f} milliseconds")
    return True, actual_ts


def find_last_valid_timestamp(
    cap: cv2.VideoCapture,
    lower_ms: float,
    upper_ms: float,
    *,
    epsilon_ms: float = 1.0,
    min_step_ms: float = 0.5,
    max_iters: int = 50
) -> float:
    """
    Find the last valid timestamp in a video using double-seek probing.

    Parameters
    ----------
    cap : cv2.VideoCapture
        OpenCV video capture object.
    lower_ms : float
        Known-good lower bound timestamp (ms). The search will start from here.
    upper_ms : float
        Known-bad upper bound timestamp (ms). The search will not go beyond this point.
    epsilon_ms : float, optional
        Small forward step for double-seek probing (default 1.0 ms).
    min_step_ms : float, optional
        Minimum search step for termination (default 0.5 ms).
    max_iters : int, optional
        Maximum number of binary search iterations to prevent infinite loops (default 50).

    Returns
    -------
    last_valid_ts : float
        Last valid timestamp in milliseconds that can be decoded by OpenCV.

    Raises
    ------
    ValueError
        If `lower_ms >= upper_ms`.
    RuntimeError
        If the lower bound is not actually readable.

    Notes
    -----
    This function is robust against:
    - Duplicate-frame plateaus at the end of a video.
    - Variable frame rate (VFR) videos.
    - Soft failures from OpenCV where `.read()` may succeed but return a repeated last frame.
    """
    if lower_ms >= upper_ms:
        raise ValueError("lower_ms must be less than upper_ms")

    # Confirm lower bound is valid
    ok, ts = seek_and_get_ts(cap, lower_ms)
    if not ok:
        raise RuntimeError("lower_ms is not actually readable")
    last_good_ts = ts

    for _ in range(max_iters):
        logger.info(f"seeking for last valid frame, lowerbound: {lower_ms:.3f}, upperbound: {upper_ms:.3f}")
        if upper_ms - lower_ms <= min_step_ms:
            break

        mid = (lower_ms + upper_ms) / 2.0

        # First seek
        ok1, ts1 = seek_and_get_ts(cap, mid)
        if not ok1:
            upper_ms = mid
            continue

        # Valid progress → mid is good
        last_good_ts = ts1
        lower_ms = mid

    return last_good_ts

