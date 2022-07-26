import json
import pandas as pd
from os.path import join as path_join
import argparse

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

def save_annotation_data(path_to_annotation, path_to_save_parsed_annotation, prefix):
    annotations = get_annotations(path_to_annotation)
    video_df, segment_df = get_annotated_videos(annotations)

    # Classify each subset of videos
    training_df = video_df[video_df['subset'] == 'training']
    validation_df = video_df[video_df['subset'] == 'validation']
    test_df = video_df[video_df['subset'] == 'testing']

    # Save them as csvs
    training_df.to_csv(path_join(path_to_save_parsed_annotation, f"{prefix}_training.csv"), index=False)
    validation_df.to_csv(path_join(path_to_save_parsed_annotation, f"{prefix}_validation.csv"), index=False)
    test_df.to_csv(path_join(path_to_save_parsed_annotation, f"{prefix}_test.csv"), index=False)
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

def convert_seconds_to_frame_indices_in_segments(to_fps = 8.0, segment_df = None, df = None):
    if segment_df is None:
        segment_df = pd.read_csv(path_join('annotations', 'annotation_data_segments.csv'))
    if df is None:
        df = pd.read_csv(path_join('annotations', 'annotation_data_training.csv'))
    elif type(df) == str:
        if df == 'train':
            df = pd.read_csv(path_join('annotations', 'annotation_data_training.csv'))
        elif df == 'valid':
            df = pd.read_csv(path_join('annotations', 'annotation_data_validation.csv'))
    
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