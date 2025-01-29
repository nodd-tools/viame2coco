import pycocowriter
import csv
import json
import itertools
from collections.abc import Iterable
import pycocowriter.coco
from pycocowriter.csv2coco import Iterable2COCO, Iterable2COCOConfig
from pycocowriter.coco import COCOLicense, COCOInfo, COCOData
from .viame_manual_annotations import *

COCO_CC0_LICENSE = COCOLicense(
    'CC0 1.0 Universal',
    0,
    'https://creativecommons.org/public-domain/cc0/'
)

viame_csv_config = {
    'filename': 1,
    'label': 9, 
    'bbox_tlbr': {
        'tlx': 2,
        'tly': 3,
        'brx': 4,
        'bry': 5
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
        video_frame_outfile_dir: str | None = None) -> tuple[
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
        data = skip_viame_metadata_rows(reader)
        if video_file is not None:
            data = extract_viame_video_annotations(
                data, video_file, outfile_dir=video_frame_outfile_dir
            )
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
        license: pycocowriter.coco.COCOLicense = COCO_CC0_LICENSE, 
        version: str = '0.1') -> pycocowriter.coco.COCOData:
    '''
    Convert a VIAME-style annotation csv into COCO format

    Parameters
    ----------
    viame_csv_file: str
        the file path location for the VIAME-style annotation csv
    descriptions: str
        the description of this dataset
    video_file: str | None
        the file path location for the video which has been
        annotated.  If there is no video (i.e. the annotations
        are for images), then this should be None
    video_frame_outfile_dir: str | None
        a directory to which the extracted frames are writ
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

    #TODO probably should hoist this into a higher function
    csv_location = os.path.split(viame_csv_file)[0]
    if video_frame_outfile_dir is None:
        video_frame_outfile_dir = csv_location
    images, annotations, categories = viame2coco_data(
        viame_csv_file, video_file=video_file, 
        video_frame_outfile_dir=video_frame_outfile_dir
    )

    return COCOData(
        coco_info, 
        images, 
        annotations, 
        [license], 
        categories
    )
