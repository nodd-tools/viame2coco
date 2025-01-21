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
    'meta': {
        'skiprows': 2
    },
    'filename': 1,
    'label': 9, 
    'bbox_tlbr': {
        'tlx': 2,
        'tly': 3,
        'brx': 4,
        'bry': 5
    }
}

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
        if video_file is not None:
            reader = itertools.chain(
                passrows(reader, 2),
                extract_viame_video_annotations(
                    reader, video_file, outfile_dir=video_frame_outfile_dir
                )
            )
        csv2coco = Iterable2COCO(
            Iterable2COCOConfig(viame_csv_config)
        )
        images, annotations, categories = csv2coco.parse(reader)
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
