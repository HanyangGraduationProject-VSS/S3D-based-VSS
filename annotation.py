import json
import pandas as pd
from os.path import join as path_join
import argparse
import cv2
from tqdm import tqdm
from config import DATASET_DIRPATH
import os

# Sample structure of annotation_data
# {
#     'duration': 25.682,
#     'subset': 'testing',
#     'resolution': '854x480',
#     'url': 'https://www.youtube.com/watch?v=aVu2j7JbYgk',
#     'annotations': [
#       {'segment': [4.957439897615345, 15.956764915336274], 'label': 'Bullfighting'},
#       {'segment': [17.815805763401784, 29.434811063811217], 'label': 'Bullfighting'},
#       {'segment': [32.84305261859799, 44.15221777766317], 'label': 'Bullfighting'},
#       {'segment': [51.27854028618512, 59.64422398136396], 'label': 'Bullfighting'}
#     ]
# }

def get_annotations(path_to_annotation):
    """
    Get annotations from an annotation file
    """
    annotations = None
    with open(path_to_annotation) as file_annotations:
        annotations = json.load(file_annotations)
    return annotations

def get_taxonomy(annotions):
    """
    Get taxonomy from annotations
    """
    taxonomy = []
    # keys: nodeId, nodeName, parentId, parentName
    for node in annotions['taxonomy']:
        taxonomy.append((node['nodeId'], node['nodeName'], node['parentId'], node['parentName']))
    return taxonomy

def get_annotation_version(annotations) -> str:
    return annotations['version']

def get_annotated_videos(annotations):
    """
    Get annotated videos from annotations
    """
    video_db = annotations['database']
    
    videos = []
    segments = []

    for video_key in video_db:
        video = video_db[video_key]
        annotation_record_id = len(segments)
        num_segments = len(video['annotations'])
        if num_segments > 0:
            segments.extend((
                segment['label'],
                segment['segment'][0],
                segment['segment'][1]
            ) for segment in video['annotations'])
        videos.append((
            video_key,
            video['duration'],
            video['subset'],
            video['resolution'],
            video['url'],
            annotation_record_id,
            num_segments
        ))

    video_df = pd.DataFrame(videos, columns=['video_id', 'duration', 'subset', 'resolution', 'url', 'annotation_record_id', 'num_segments'])
    segment_df = pd.DataFrame(segments, columns=['label', 'start_time', 'end_time'])

    return video_df, segment_df

def fetch_video_metatdata(video_path, video_name = None):
    if video_name is None:
        video_name = video_path.split("/")[-1]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
      return None
    else:
      ret = (
          str(video_name),
          int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
          int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
          float(cap.get(cv2.CAP_PROP_FPS))
      )
      cap.release()
      return ret

def get_video_info(video_files):
  for video_file_entry in tqdm(video_files):
      ret = fetch_video_metatdata(video_file_entry.path, video_file_entry.name)
      if ret:
        yield ret
      else:
        print(f"{video_file_entry.name} cannot be open")

def save_annotation_data(path_to_annotation, path_to_save_parsed_annotation, prefix):
    annotations = get_annotations(path_to_annotation)
    video_df, segment_df = get_annotated_videos(annotations)

    # Classify each subset of videos
    training_df = video_df[video_df['subset'] == 'training']
    validation_df = video_df[video_df['subset'] == 'validation']
    test_df = video_df[video_df['subset'] == 'testing']

    cond = lambda item: item.is_file() and item.name.endswith('.mp4') or item.name.endswith('.mkv') or item.name.endswith('.webm')
    
    train_val_metatdata_df = None
    test_metadata_df = None

    train_val_dir_path = path_join(DATASET_DIRPATH, 'train_val')
    if os.path.exists(train_val_dir_path):
        train_val_video_files = [item for item in os.scandir(train_val_dir_path) if cond(item)]

        if len(train_val_video_files) == 0:
            print('no video files found in train_val directory')
        else:
            train_val_metatdata_df = pd.DataFrame(get_video_info(train_val_video_files), columns = ["video_name", "width", "height", "total_frames", "fps"])
            train_val_metatdata_df = train_val_metatdata_df.convert_dtypes()
            train_val_metatdata_df.video_name = train_val_metatdata_df.video_name.apply(lambda name: str(name).split('.')[0][2:])
            train_val_metatdata_df.rename(columns = {'video_name':'video_id'}, inplace=True)
    else:
        print('train & validation set directory does not exist')

    test_dir_path = path_join(DATASET_DIRPATH, 'test')
    if os.path.exists(test_dir_path):
        test_video_files = [item for item in os.scandir(test_dir_path) if cond(item)]

        if len(test_video_files) == 0:
            print('no video files found in test directory')
        else:
            test_metadata_df = pd.DataFrame(get_video_info(test_video_files), columns = ["video_name", "width", "height", "total_frames", "fps"])
            test_metadata_df = test_metadata_df.convert_dtypes()
            test_metadata_df.video_name = test_metadata_df.video_name.apply(lambda name: str(name).split('.')[0][2:])
            test_metadata_df.rename(columns = {'video_name':'video_id'}, inplace=True)
    else:
        print('test set directory does not exist')

    # Save them as csvs
    common_columns = ["video_id", "duration", "fps", "total_frames",
                      "width", "height", "url", "annotation_record_id", "num_segments"]

    if train_val_metatdata_df is not None:
        pd.merge(train_val_metatdata_df, training_df, on='video_id')[common_columns].to_csv(path_join(path_to_save_parsed_annotation, f"{prefix}_training.csv"), index=False)
        pd.merge(train_val_metatdata_df, validation_df, on='video_id')[common_columns].to_csv(path_join(path_to_save_parsed_annotation, f"{prefix}_validation.csv"), index=False)

    if test_metadata_df is not None:
        pd.merge(test_metadata_df, test_df, on='video_id')[common_columns].to_csv(path_join(path_to_save_parsed_annotation, f"{prefix}_test.csv"), index=False)
    
    segment_df.to_csv(path_join(path_to_save_parsed_annotation, f"{prefix}_segments.csv"), index=False)

def load_annotation_data(path_to_annotation, prefix):
    training_df = pd.read_csv(path_join(path_to_annotation, f"{prefix}_training.csv"))
    validation_df = pd.read_csv(path_join(path_to_annotation, f"{prefix}_validation.csv"))
    test_df = pd.read_csv(path_join(path_to_annotation, f"{prefix}_test.csv"))
    segment_df = pd.read_csv(path_join(path_to_annotation, f"{prefix}_segments.csv"))

    return training_df, validation_df, test_df, segment_df

def describe_annotation_data(training_df, validation_df, test_df, segment_df):
    print('Training:')
    print(training_df.describe())
    print()
    
    print('Validation:')
    print(validation_df.describe())
    print()

    print('Test:')
    print(test_df.describe())
    print()

    print('Segments:')
    print(segment_df.describe())
    print()

def show_segment_distributions(annotation_dirpath, prefix):
    segment_df = pd.read_csv(path_join(annotation_dirpath, f"{prefix}_segments.csv"))
    print(segment_df['label'].value_counts().head(20))

def get_annotations_dataframe(dataset_name = None):
    if dataset_name is None:
        return pd.read_csv(path_join('annotations', 'annotation_data_training.csv'))
    elif type(dataset_name) == str:
        if dataset_name == 'train':
            return pd.read_csv(path_join('annotations', 'annotation_data_training.csv'))
        elif dataset_name == 'valid':
            return pd.read_csv(path_join('annotations', 'annotation_data_validation.csv'))
        elif dataset_name == 'test':
            return pd.read_csv(path_join('annotations', 'annotation_data_test.csv'))
    elif type(dataset_name) == pd.DataFrame:
        return dataset_name
    assert False, "Invalid dataset name"

def convert_seconds_to_frame_indices_in_segments(to_fps = 8.0, segment_df = None, df = None):
    if segment_df is None:
        segment_df = pd.read_csv(path_join('annotations', 'annotation_data_segments.csv'))
    
    df = get_annotations_dataframe(df)
    
    new_df =  pd.DataFrame((
        (*v[:-2], *s) for v in df.values for s in segment_df.iloc[v[-2]:v[-2]+v[-1]].values
    ), columns=list(df.columns[:-2])+list(segment_df.columns))

    new_df["start_frame"] = new_df["start_time"] * new_df["fps"] / to_fps
    new_df["end_frame"] = new_df["end_time"] * new_df["fps"] / to_fps
    new_df["start_frame"] = new_df["start_frame"].apply(round, int)
    new_df["end_frame"] = new_df["end_frame"].apply(round, int)

    new_df.set_index("video_id", inplace=True)

    return new_df

def main():
    default_annotation_dirpath = path_join('.', 'annotations')
    default_annotation_path = path_join('.', 'annotations', 'activity_net.v1-3.min.json')
    annotation_prefix = 'annotation_data'

    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, default=default_annotation_path, help='full path of the annotation json file')
    parser.add_argument('--out', type=str, default=default_annotation_dirpath, help='full path of CSV annotation files')
    args = parser.parse_args()

    save_annotation_data(args.json, args.out, annotation_prefix)
    describe_annotation_data(*load_annotation_data(args.out, annotation_prefix))

if __name__ == '__main__':
    main()