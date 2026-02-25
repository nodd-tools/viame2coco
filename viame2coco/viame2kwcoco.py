import csv
import kwcoco
import os
import logging
import argparse

# Configure module-level logger
logger = logging.getLogger(__name__)

def parse_viame_row(row):
    """
    Extract standard VIAME columns and the top ragged-right class from a CSV row.

    Parameters
    ----------
    row : list of str
        A single row from a VIAME CSV file, representing one annotation.

    Returns
    -------
    tuple
        A tuple containing:
        - track_id (int): The tracking ID (-1 if untracked).
        - image_name (str): The filename of the image/frame.
        - frame_id (int): The integer frame index.
        - bbox (list of float): The bounding box in [x, y, width, height] format.
        - best_class_name (str): The class name with the highest confidence score.
        - highest_score (float): The highest confidence score.
    """
    track_id = int(row[0])
    image_name = row[1]
    frame_id = int(row[2])
    
    x1, y1 = float(row[3]), float(row[4])
    x2, y2 = float(row[5]), float(row[6])
    bbox = [x1, y1, x2 - x1, y2 - y1]
    
    best_class_name = "unknown"
    highest_score = 0.0
    
    # Parse the ragged-right classes (starts at index 9)
    for i in range(9, len(row), 2):
        if i + 1 < len(row): 
            c_name = row[i]
            c_score = float(row[i+1])
            if c_score > highest_score:
                highest_score = c_score
                best_class_name = c_name
                
    return track_id, image_name, frame_id, bbox, best_class_name, highest_score

def convert_viame_to_kwcoco(csv_path, output_json_path, video_name=None):
    """
    Convert a single VIAME CSV into a standalone kwcoco dataset.

    Parameters
    ----------
    csv_path : str
        Path to the input VIAME CSV file.
    output_json_path : str
        Path where the resulting kwcoco JSON file will be saved.
    video_name : str, optional
        Name of the video. If provided, images will be linked sequentially 
        to a video sequence. Default is None (standalone images).

    Returns
    -------
    kwcoco.CocoDataset
        The populated kwcoco dataset object.
    """
    logger.debug(f"Starting conversion for {csv_path}")
    dset = kwcoco.CocoDataset()
    
    vid_id = None
    if video_name:
        vid_id = dset.ensure_video(name=video_name)
        
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
                
            track_id, image_name, frame_id, bbox, best_class, score = parse_viame_row(row)
            
            img_kwargs = {'file_name': image_name}
            if vid_id is not None:
                img_kwargs['video_id'] = vid_id
                img_kwargs['frame_index'] = frame_id
                
            img_id = dset.ensure_image(**img_kwargs)
            cat_id = dset.ensure_category(name=best_class)
            
            ann = {
                'image_id': img_id,
                'category_id': cat_id,
                'bbox': bbox,
                'score': score,
            }
            
            if track_id >= 0: 
                # Explicitly register the track to avoid kwcoco warnings
                if track_id not in dset.index.tracks:
                    dset.add_track(id=track_id, name=str(track_id))
                ann['track_id'] = track_id
                
            dset.add_annotation(**ann)

    dset.fpath = output_json_path
    dset.dump(dset.fpath, newlines=True)
    
    n_imgs = len(dset.index.imgs)
    n_anns = len(dset.index.anns)
    logger.info(f"Saved: {output_json_path} | Images: {n_imgs} | Annotations: {n_anns}")
    
    return dset

def main():
    """
    Command-line interface for converting a VIAME CSV to kwcoco format.
    """
    parser = argparse.ArgumentParser(
        description="Convert a VIAME CSV format file to a kwcoco JSON dataset."
    )
    
    parser.add_argument(
        "input_csv", 
        help="Path to the input VIAME CSV file."
    )
    parser.add_argument(
        "output_json", 
        help="Path to save the output kwcoco JSON file."
    )
    parser.add_argument(
        "--video-name", 
        type=str, 
        default=None,
        help="Optional video name. If provided, links frames into a video sequence."
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Enable debug-level logging."
    )
    
    args = parser.parse_args()
    
    # Configure logging level based on arguments
    logging_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    convert_viame_to_kwcoco(args.input_csv, args.output_json, video_name=args.video_name)

if __name__ == "__main__":
    main()
