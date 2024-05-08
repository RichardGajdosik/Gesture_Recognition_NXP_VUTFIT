import json
import os

base_dataset_path = '/home/default/dataset/small_dataset/dataset_small_10_percent'
annotations_directory = '../dataset/annotations/train'
filtered_annotations_directory = os.path.join(base_dataset_path, 'annotations_small_10_percent')

gestures = ['dislike', 'fist', 'like', 'peace', 'stop']

if not os.path.exists(filtered_annotations_directory):
    os.makedirs(filtered_annotations_directory)

for gesture in gestures:
    gesture_images_directory = os.path.join(base_dataset_path, gesture)
    gesture_annotations_path = os.path.join(annotations_directory, f'{gesture}.json')

    with open(gesture_annotations_path, 'r') as file:
        annotations = json.load(file)

    image_files = {os.path.splitext(file)[0] for file in os.listdir(gesture_images_directory) if file.endswith('.jpg')}

    filtered_annotations = {key: value for key, value in annotations.items() if key in image_files}

    filtered_annotations_path = os.path.join(filtered_annotations_directory, f'{gesture}.json')
    with open(filtered_annotations_path, 'w') as file:
        json.dump(filtered_annotations, file, indent=4)

    print(f"Filtration for: {gesture} number of annotations: {len(filtered_annotations)}")