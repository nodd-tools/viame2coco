import pycocowriter
import csv
import json
import re
import itertools
import datetime
from collections.abc import Iterable
import pycocowriter.coco
from pycocowriter.csv2coco import Iterable2COCO, Iterable2COCOConfig
from pycocowriter.coco import COCOLicense, COCOInfo, COCOData
from .viame_manual_annotations import *
import logging

logger = logging.getLogger(__name__)

COCO_CC0_LICENSE = COCOLicense(
    'CC0 1.0 Universal',
    0,
    'https://creativecommons.org/public-domain/cc0/'
)

viame_csv_config_default = {
    'filename': 1,
    'label': 9, 
    'bbox_tlbr': {
        'tlx': 3,
        'tly': 4,
        'brx': 5,
        'bry': 6
    }
}

def is_viame_metadata_row(row: Sequence[str]) -> bool:
    '''
    determines whether this row is a "metadata" row in a viame
    csv

    Parameters
    ----------
    row: Sequence[str]
        a row read in from a VIAME-style annotation csv

    Returns
    -------
    is_metadata: bool
        true if the row arg is a metadata row
    '''
    is_metadata = row[0].startswith('#')
    return is_metadata

def skip_viame_metadata_rows(
        viame_rows: Iterable[Sequence[str]]) -> Iterable[Sequence[str]]:
    '''
    skip any metadata rows in a sequence of VIAME-style annotation rows
    as read from a VIAME output csv

    Parameters
    ----------
    viame_rows: Iterable[Sequence[str]]
        an iterable of rows as read from a VIAME-style annotation csv output

    Returns
    -------
    viame_rows: Iterable[Sequence[str]]
        the same iterable of rows, but having skipped any metadata rows
    '''
    row = next(viame_rows)
    while is_viame_metadata_row(row):
        row = next(viame_rows)
    yield row
    yield from viame_rows

def read_viame_metadata_rows(
        viame_rows: Iterable[Sequence[str]]) -> tuple[list[Sequence[str]], Iterable[Sequence[str]]]:
    '''
    skip any metadata rows in a sequence of VIAME-style annotation rows
    as read from a VIAME output csv, and return those rows along with the remainder
    of the annotations iterator.

    Parameters
    ----------
    viame_rows: Iterable[Sequence[str]]
        an iterable of rows as read from a VIAME-style annotation csv output

    Returns
    -------
    metadata_rows: list[Sequence[str]]
        the metadata rows

    viame_rows: Iterable[Sequence[str]]
        the same iterable of rows, but having skipped any metadata rows
    '''
    metadata_rows = []
    for row in viame_rows:
        if not is_viame_metadata_row(row):
            return metadata_rows, itertools.chain([row], viame_rows)
        metadata_rows.append(row)
    return metadata_rows, iter(())


V1_METADATA_HEADER_PATTERN = re.compile(
    r"^# 1: Detection or Track-id"
    r"2: Video or Image Identifier"
    r"3: Unique Frame Identifier"
    r"4-7: Img-bbox\(TL_xTL_yBR_xBR_y\)"
    r"8: Detection or Length Confidence"
    r"9: Target Length \(0 or -1 if invalid\)"
    r"10-11\+: Repeated SpeciesConfidence Pairs or Attributes"
    r"# metadata - fps: (\d+)"
    r"# Written on ([A-Za-z]{3} [A-Za-z]{3} \d{1,2} "
    r"\d{2}:\d{2}:\d{2} \d{4}) by: dive:python"
)
V2_METADATA_HEADER_PATTERN = re.compile(
    r"^# 1: Detection or Track-id"
    r"2: Video or Image Identifier"
    r"3: Unique Frame Identifier"
    r"4-7: Img-bbox\(TL_xTL_yBR_xBR_y\)"
    r"8: Detection or Length Confidence"
    r"9: Target Length \(0 or -1 if invalid\)"
    r"10-11\+: Repeated SpeciesConfidence Pairs or Attributes"
    r"# metadatafps: (\d+)"
    r'exported_by: "dive:typescript"'
    r'exported_time: "(\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2}:\d{2} [AP]M)"'
)
V3_METADATA_HEADER_PATTERN = re.compile(
    r"^# 1: Detection or Track-id"
    r"2: Video or Image Identifier"
    r"3: Unique Frame Identifier"
    r"4-7: Img-bbox\(TL_xTL_yBR_xBR_y\)"
    r"8: Detection or Length Confidence"
    r"9: Target Length \(0 or -1 if invalid\)"
    r"10-11\+: Repeated SpeciesConfidence Pairs or Attributes"
    r"#meta fps=(\d+)"
    r"# Written on (\d{1,2}/\d{1,2}/\d{4} "
    r"\d{1,2}:\d{2}:\d{2} [AP]M) by dive_writer:typescript"
)
V4_METADATA_HEADER_PATTERN = re.compile(
    r'^# 1: Detection or Track-id'
    r'2: Video or Image Identifier'
    r'3: Unique Frame Identifier'
    r'4-7: Img-bbox\(TL_xTL_yBR_xBR_y\)'
    r'8: Detection or Length Confidence'
    r'9: Target Length \(0 or -1 if invalid\)'
    r'10-11\+: Repeated SpeciesConfidence Pairs or Attributes'
    r'# metadata'
    r'fps: (\d+)'
    r'exported_by: "dive:python"'
    r'exported_time: "([A-Za-z]{3} [A-Za-z]{3} \d{1,2} \d{2}:\d{2}:\d{2} \d{4})"$'
)
V5_METADATA_HEADER_PATTERN = re.compile(
    r'^# 1: Detection or Track-id'
    r'2: Video or Image Identifier'
    r'3: Unique Frame Identifier'
    r'4-7: Img-bbox\(TL_xTL_yBR_xBR_y\)'
    r'8: Detection or Length Confidence'
    r'9: Target Length \(0 or -1 if invalid\)'
    r'10-11\+: Repeated SpeciesConfidence Pairs or Attributes'
    r'# metadata'
    r'fps:\s*([0-9]*\.?[0-9]+)'
    r'exported_by: "dive:typescript"'
    r'exported_time: "(\d{1,2}/\d{1,2}/\d{4}, '
    r'\d{1,2}:\d{2}:\d{2} [AP]M)"$'
)
V6_METADATA_HEADER_PATTERN = re.compile(
    r'^# 1: Detection or Track-id'
    r'2: Video or Image Identifier'
    r'3: Unique Frame Identifier'
    r'4-7: Img-bbox\(TL_xTL_yBR_xBR_y\)'
    r'8: Detection or Length Confidence'
    r'9: Target Length \(0 or -1 if invalid\)'
    r'10-11\+: Repeated SpeciesConfidence Pairs or Attributes'
    r'# metadata - fps: ([0-9]+)'
    r'# Written on ([A-Za-z]{3} [A-Za-z]{3} '
    r'\d{1,2} \d{2}:\d{2}:\d{2} \d{4}) by: '
    r'viame_web_csv_writer:python$'
)

def determine_viame_version(viame_metadata_rows: list[Sequence[str]]) -> int:
    '''
    Determine the viame "version" from the metadata rows.
    We need these to figure out how to parse the remainder of the file

    It's not 100% clear that we can determine the version just from the metadata.
    Fingers crossed.

    Parameters
    ----------
    viame_metadata_rows: list[Sequence[str]]
        rows read by `read_viame_metadata_rows`

    Returns
    ----------
    version: int
        a pseudo-version of VIAME that we can use to parse the rest of the file

    fps: int | None
        the framerate of the annotations, read from the metadata or None if no fps present

    timestamp: datetime.datetime | None
        the date/time stamp of the annotations, read from the metadata or None if no timestamp present
    '''
    version, fps, timestamp = None, None, None
    raw_metadata = ''.join(map(''.join, viame_metadata_rows))
    logger.info(raw_metadata)
    if m := V1_METADATA_HEADER_PATTERN.fullmatch(raw_metadata):
        version = 1
        fps = float(m.group(1))
        timestamp = datetime.datetime.strptime(m.group(2), "%a %b %d %H:%M:%S %Y")
    elif m := V2_METADATA_HEADER_PATTERN.fullmatch(raw_metadata):
        version = 2
        fps = float(m.group(1))
        timestamp = datetime.datetime.strptime(m.group(2), "%m/%d/%Y, %I:%M:%S %p")
    elif m := V3_METADATA_HEADER_PATTERN.fullmatch(raw_metadata):
        version = 3
        fps = float(m.group(1))
        timestamp = datetime.datetime.strptime(m.group(2), "%m/%d/%Y %I:%M:%S %p")
    elif m := V4_METADATA_HEADER_PATTERN.fullmatch(raw_metadata):
        version = 4
        fps = float(m.group(1))
        timestamp = datetime.datetime.strptime(m.group(2), "%a %b %d %H:%M:%S %Y")
    elif m := V5_METADATA_HEADER_PATTERN.fullmatch(raw_metadata):
        version = 5
        fps = float(m.group(1))
        timestamp = datetime.datetime.strptime(m.group(2), "%m/%d/%Y, %I:%M:%S %p")
    elif m := V6_METADATA_HEADER_PATTERN.fullmatch(raw_metadata):
        version = 6
        fps = float(m.group(1))
        timestamp = datetime.datetime.strptime(m.group(2), "%a %b %d %H:%M:%S %Y")
    return version, fps, timestamp


def deal_with_viame_timestamps(viame_rows: Iterable[Sequence[str]], version: int, fps: int) -> Iterable[Sequence[str]]:
    '''
    The frame timestamps produced by VIAME are frequently wrong.
    Unfortunately, it is difficult to tell the difference, so we
    just have to not rely on this column, and compute the timestamp from
    the frame number and framerate.
    
    Parameters
    ----------
    viame_rows: Iterable[Sequence[str]]
        The actual data rows from the viame csv

    version: int
        a data format version number as determined by `determine_viame_version`

    fps: int
        the fps read from the metadata headers

    Returns
    ---------
    Iterable[Sequence[str]]
        A "fixed" set of data with isotimestamps that the rest of the processing
        expects.
    '''
    '''
    for row in viame_rows:
        try:
            dt = datetime.time.fromisoformat(row[1]) # maybe it's in isoformat, this is fine, leave it be
            logger.debug(f'read timestamp: {row[1]}, result: {dt.isoformat()}')
            yield row
        except ValueError as e:
            # probably it's garbage.  Compute a new timestamp column from fps and frame
            seconds = int(row[2]) / fps
            timestamp = (datetime.datetime.min + datetime.timedelta(seconds=seconds)).time()
            new_row1 = timestamp.isoformat()
            logger.debug(f'recovering timestamp err: {row[1]}, fps: {fps}, frame: {row[2]}, result: {new_row1}')
            row[1] = new_row1
            yield row
    '''
    for row in viame_rows:
        seconds = int(row[2]) / fps
        timestamp = (datetime.datetime.min + datetime.timedelta(seconds=seconds)).time()
        new_row1 = timestamp.isoformat()
        logger.debug(f'recovering timestamp err: {row[1]}, fps: {fps}, frame: {row[2]}, result: {new_row1}')
        row[1] = new_row1
        yield row

def passrows(iterable: Iterable, n: int = 0) -> Iterable:
    '''
    yield the first `n` rows in `iterable`.
    Useful with `itertools.chain` and `map` to 
    apply a function to only certain rows of an iterable

    Parameters
    ----------
    iterable: Iterable
        any iterable
    n: int
        the number of rows to skip

    Returns
    -------
    iterable: Iterable
        the iterable arg, but starting from the n+1th row
    '''
    for i in range(n):
        yield next(iterable)

def viame2coco_data(
        viame_csv_file: str, 
        video_file: str | None = None, 
        video_frame_outfile_dir: str | None = None, 
        viame_csv_config : dict | None = None) -> tuple[
            list[pycocowriter.coco.COCOImage],
            list[pycocowriter.coco.COCOAnnotation],
            list[pycocowriter.coco.COCOCategory]
        ]:
    '''
    extract the images, annotations, and categories from a VIAME-style
    annotation csv, into COCO format.  Filters the data to only MANUAL
    annotations.

    If the annotations are for a video file, also extract the images
    for the manually-annotated frames

    Parameters
    ----------
    viame_csv_file: str
        the file path location for the VIAME-style annotation csv
    video_file: str | None
        the file path location for the video which has been
        annotated.  If there is no video (i.e. the annotations
        are for images), then this should be None
    video_frame_outfile_dir: str | None
        a directory to which the extracted frames are writ
    viame_csv_config : dict | None
        the dictionary that specifies which fields are present, 
        and in which columns they are located. 
        Passed to pycocowriter.csv2coco.Iterable2COCOConfig. 
        If None, then viame2coco.viame2coco.viame_csv_config is used
    
    Returns
    -------
    images: list[COCOImage]
        a list of images contained in the CSV file, in COCO format,
        with appropriately-generated surrogate keys
    annotations: list[COCOAnnotation]
        a list of the annotations contained in the CSV file, with
        appropriate surrogate-key references to the images and categories
    categories: list[COCOCategory]
        a list of the categories contained in the CSV file, in COCO format,
        with appropriately-generated surrogate keys
    '''
    with open(viame_csv_file, 'r') as f:
        reader = csv.reader(f)
        metadata, data = read_viame_metadata_rows(reader)
        viame_version, fps, timestamp = determine_viame_version(metadata)
        data = deal_with_viame_timestamps(data, viame_version, fps)
        if video_file is not None:            
            #TODO probably should hoist this into a higher function            
            if video_frame_outfile_dir is None:
                csv_location = os.path.split(viame_csv_file)[0]
                video_frame_outfile_dir = csv_location
            data = extract_viame_video_annotations(
                data, video_file, outfile_dir=video_frame_outfile_dir
            )
        if viame_csv_config is None:
            viame_csv_config = viame_csv_config_default
        csv2coco = Iterable2COCO(
            Iterable2COCOConfig(viame_csv_config)
        )
        images, annotations, categories = csv2coco.parse(data)
        return images, annotations, categories

def viame2coco(
        viame_csv_file: str, 
        description: str, 
        video_file: str | None = None, 
        video_frame_outfile_dir: str | None = None,
        viame_csv_config : dict | None = None, 
        license: pycocowriter.coco.COCOLicense = COCO_CC0_LICENSE, 
        version: str = '0.1') -> pycocowriter.coco.COCOData:
    '''
    Convert a VIAME-style annotation csv into COCO format

    Parameters
    ----------
    viame_csv_file: str
        the file path location for the VIAME-style annotation csv
    description: str
        the description of this dataset
    video_file: str | None
        the file path location for the video which has been
        annotated.  If there is no video (i.e. the annotations
        are for images), then this should be None
    video_frame_outfile_dir: str | None
        a directory to which the extracted frames are writ
    viame_csv_config : dict | None
        the dictionary that specifies which fields are present, 
        and in which columns they are located. 
        If None, then viame2coco.viame2coco.viame_csv_config is used
    license: COCOLicense
        the license under which these images are provided
        Defaults to CC0 https://creativecommons.org/public-domain/cc0/
    version: str
        the version of this dataset, as a string
        defaults to '0.1'
    '''
    
    now = datetime.datetime.now(datetime.timezone.utc)
    coco_info = COCOInfo(
        year = now.year,
        version = version, 
        description = description, 
        date_created = now
    )
    logger.info(f"converting video {video_file} and annotations {viame_csv_file}")
    images, annotations, categories = viame2coco_data(
        viame_csv_file, video_file=video_file, 
        video_frame_outfile_dir=video_frame_outfile_dir, 
        viame_csv_config = viame_csv_config
    )

    return COCOData(
        coco_info, 
        images, 
        annotations, 
        [license], 
        categories
    )
