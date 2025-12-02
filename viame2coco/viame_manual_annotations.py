import cv2
import os
import datetime
import numpy as np
from collections.abc import Sequence, Iterable
import logging
from .vid_utils import find_last_valid_timestamp

logger = logging.getLogger(__name__)

MS2S = 1000000
MS2M = MS2S*60
MS2H = MS2M*60
def time2micros(time: datetime.time) -> float:
    '''
    convert a datetime.time into total microseconds

    ```
    >>> time2micros(datetime.time(1,1,1)) # 1 hour, 1 min, 1 sec
    3661000000
 
    ```

    Parameters
    ----------
    time: datetime.time
        the time to convert into microseconds

    Returns
    -------
    microseconds: float | int
        the total number of microseconds in the time argument        
    '''
    return time.hour * MS2H + time.minute * MS2M + time.second * MS2S + time.microsecond

def extract_frame_microseconds(
        cv2_video_cap: cv2.VideoCapture, 
        microseconds: float, 
        outfile: str | None = None) -> np.ndarray | None:
    '''
    extract a frame from the provided cv2 video at the given number 
    of microseconds.  Optionally write the frame to outfile.

    Parameters
    ----------
    cv2_video_cap: cv2.VideoCapture
        the video from which to capture the frame
    microseconds: float
        the location in microseconds into the video
        at which to extract the desired frame
    outfile: str | None:
        the optional filename to which the desired frame should 
        be writ
    Returns
    -------
    image: numpy.ndarray | None
        the video frame at the given number of microseconds, or None
        if the frame read was unsuccessful.  Additionally, the frame
        may be written to a file as a side-effect if `outfile` was
        passed as an argument.
    '''
    logger.debug(f"extracting frame at {microseconds:.3f} microseconds")
    cv2_video_cap.set(cv2.CAP_PROP_POS_MSEC, microseconds // 1000)
    success, image = cv2_video_cap.read()
    if outfile is not None:
        try:
            cv2.imwrite(outfile, image)
        except cv2.error as e:
            # sometimes times very close to the end are "too far", scale it back to the end
            ALLOWED_FUDGE_MICROS = 10000 # ten milliseconds
            PROBLEMATIC_DATA_MICROS = 1000000 # one second
            logger.info(f"issue reading video at {microseconds:.3f} microseconds")
            last_valid_timestamp = find_last_valid_timestamp(cv2_video_cap, 0, microseconds / 1000) # in milliseconds
            logger.info(f"last valid timestamp found at {last_valid_timestamp:.3f} milliseconds")
            fudge = microseconds/1000 - last_valid_timestamp # milliseconds
            logger.info("fudging {} milliseconds".format(fudge))
            if fudge > PROBLEMATIC_DATA_MICROS:
                raise Exception(f"Something is wrong with this data, annotation is {fudge:.3f} ms away from end of video at {last_valid_timestamp:.3f}")
            elif fudge < ALLOWED_FUDGE_MICROS:
                # this is fine, just use the last frame
                logger.info(f"fudging annotation at {fudge:.3f} ms from computed end of video")
                cv2_video_cap.set(cv2.CAP_PROP_POS_MSEC, last_valid_timestamp)
                success, image = cv2_video_cap.read()
                if outfile is not None:
                    # if this still fails, let the error bubble up
                    cv2.imwrite(outfile, image)
            else:
                # timestamp is outside of allowed fudge factor (frame will be too far from annotation),
                # but not so far as to indicate problematic data.  Just ditch this datum and move on.
                logger.info(f"discarding annotation at {fudge:.3f} ms from computed end of video")
                return None
    return image

VIAME_CONFIDENCE_COL = 7
def viame_is_manual_annotation(viame_csv_row: Sequence) -> bool:
    '''
    returns whether a given row in a VIAME-style annotation output csv
    represents a manual annotation or an automated annotation.

    basically, just checks if the annotation confidence is 1

    Parameters
    ----------
    viame_csv_row: Sequence
        a row read from a VIAME-style annotation csv

    Returns
    -------
    is_manual_annotation: bool
        a boolean representing whether this row is manual or not 
    '''
    is_manual_annotation = (
        (len(viame_csv_row) > VIAME_CONFIDENCE_COL) 
            and 
        (float(viame_csv_row[VIAME_CONFIDENCE_COL]) == 1)
    )
    return is_manual_annotation

def construct_image_filename_from_video_frame(
        video_filename: str, 
        time: datetime.time, 
        outfile_format: str | None, 
        outfile_dir: str | None) -> str:
    '''
    construct a filename from a given video file and frame time

    Parameters
    ----------
    video_filename: str
        the file name of the video.  This will be formatted into the
        outfile_format as `video_filename`.  See that arg for more details.
    time: datetime.time
        the time that locates the desired frame in the video
    outfile_format: str | None
        if None, defaults to '{video_filename}.%H.%M.%S.%f.jpg'
        `video_filename` is the argument of this name to this function
        The remainder is passed through a `strftime` from the time arg,
        see the [`strftime` docs](https://docs.python.org/3/library/datetime.html#format-codes)
        the extension `.jpg` will determine the output file format if this
        filename is used to write an image file.
    outfile_dir: str | None
        if not None, this is simply path joined to the filename output

    Returns
    -------
    frame_filename: str
        a filename appropriate for the specified frame in the video
    '''
    if outfile_format is None:
        outfile_format = '{video_filename}.%H.%M.%S.%f.jpg'
    frame_filename = time.strftime(outfile_format).format(video_filename = video_filename)
    if outfile_dir is not None:
        frame_filename = os.path.join(outfile_dir, frame_filename)
    return frame_filename

def filter_viame_manual_annotations(
        viame_csv: Iterable[Sequence]) -> Iterable[Sequence]:
    '''
    filters an iterable of data rows read from a VIAME-style annotation csv
    to only rows that contain manual annotations

    Parameters
    ----------
    viame_csv: Iterable[Sequence]
        the data rows from a VIAME-style annotation csv
        should not include the headers
    
    Returns
    -------
    viame_csv: Iterable[Sequence]
        the data rows in the input only when the annotations
        are manual, skipping any automated annotations
    '''
    yield from filter(viame_is_manual_annotation, viame_csv)

VIAME_VIDEO_TIME_COL = 1
def extract_viame_video_annotations(
        viame_csv: Iterable[Sequence], 
        video_file: str, 
        outfile_format: str | None = None, 
        outfile_dir: str | None = None) -> Iterable[Sequence]:
    '''
    extract the manual annotations and frames from a VIAME-style
    annotaiton csv

    Writes the frames to files.

    Parameters
    ----------
    viame_csv: Iterable[Sequence]
        the data rows from a VIAME-style annotation csv
        should not include the headers
    video_file: str
        the file name of the video.  This will be formatted into the
        outfile_format as `video_filename`.  See that arg for more details.
    outfile_format: str | None
        see `construct_image_filename_from_video_frame` signature
    outfile_dir: str | None
        see `construct_image_filename_from_video_frame` signature

    Returns
    -------
    viame_csv: Iterable[Sequence]
        the data rows in the input only when the annotations
        are manual, skipping any automated annotations
    '''
    logger.info(f"extracting images from video {video_file}")
    cap = cv2.VideoCapture(video_file)
    logger.info("Video opened: %s", cap.isOpened())
    logger.info("Frame count: %s", cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("FPS: %s", cap.get(cv2.CAP_PROP_FPS))
    logger.info("Duration (ms, naive): %s", cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS)*1000 if cap.get(cv2.CAP_PROP_FPS) else "unknown")
    video_filename_leaf = os.path.split(video_file)[1]
    if outfile_dir is not None:
        os.makedirs(outfile_dir, exist_ok=True)
    for row in filter_viame_manual_annotations(viame_csv):
        frame_time = datetime.time.fromisoformat(row[VIAME_VIDEO_TIME_COL])
        microseconds = time2micros(frame_time)
        frame_filename = construct_image_filename_from_video_frame(video_filename_leaf, frame_time, outfile_format, outfile_dir)
        image = extract_frame_microseconds(cap, microseconds, frame_filename)
        if image is not None:
            # if we fail to extract the image, just move along
            row[VIAME_VIDEO_TIME_COL] = frame_filename
            yield row
        else:
            logger.info(f"image extraction failed {frame_filename}")
