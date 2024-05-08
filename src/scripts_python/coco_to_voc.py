import json
import os
from xml.etree.ElementTree import Element, SubElement, ElementTree
from PIL import Image

def create_voc_xml(image_id, annotations, image_dimensions, output_directory):
    print(f'Rozmery obrázka {image_id}.jpg: {image_dimensions}')
    root = Element('annotation')
    SubElement(root, 'filename').text = image_id + '.jpg'

    source = SubElement(root, 'source')
    SubElement(source, 'database').text = "Unknown"

    size = SubElement(root, 'size')
    SubElement(size, 'width').text = str(image_dimensions[0])
    SubElement(size, 'height').text = str(image_dimensions[1])
    SubElement(size, 'depth').text = str(image_dimensions[2])

    SubElement(root, 'segmented').text = "0"

    for annotation in annotations:
        obj = SubElement(root, 'object')
        SubElement(obj, 'name').text = annotation['label']
        SubElement(obj, 'pose').text = "Unspecified"
        SubElement(obj, 'truncated').text = "0"
        SubElement(obj, 'difficult').text = "0"
        bndbox = SubElement(obj, 'bndbox')
        SubElement(bndbox, 'xmin').text = str(annotation['bbox'][0])
        SubElement(bndbox, 'ymin').text = str(annotation['bbox'][1])
        SubElement(bndbox, 'xmax').text = str(annotation['bbox'][0] + annotation['bbox'][2])
        SubElement(bndbox, 'ymax').text = str(annotation['bbox'][1] + annotation['bbox'][3])
    
    tree = ElementTree(root)
    tree.write(os.path.join(output_directory, f'{image_id}.xml'))

base_dataset_path = '/home/default/dataset/dataset_5_gestures/dataset_small_10_percent_combined_in_one_dir'
images_directory = os.path.join(base_dataset_path, 'dataset')
annotations_directory = os.path.join(base_dataset_path, 'annotations_small_10_percent')
voc_output_directory = os.path.join(base_dataset_path, 'annotations_small_10_percent_VOC')

gestures = ['dislike', 'fist', 'like', 'peace', 'stop']

print(f'Kontrolujem existenciu výstupného adresára: {voc_output_directory}')
if not os.path.exists(voc_output_directory):
    print(f'Výstupný adresár vytvorený: {voc_output_directory}')
    os.makedirs(voc_output_directory)

# Convert coco to voc for each gesture
for gesture in gestures:
    annotation_file_path = os.path.join(annotations_directory, f'{gesture}.json')
    if not os.path.isfile(annotation_file_path):
        print(f'Súbor s anotáciami neexistuje: {annotation_file_path}, preskakujem.')
        continue
    print(f'Načítavam anotácie z: {annotation_file_path}')
    with open(annotation_file_path, 'r') as file:
        annotations = json.load(file)
    
    for image_id, annotation in annotations.items():
        image_path = os.path.join(images_directory, f'{image_id}.jpg')
        print(f'Kontrolujem existenciu obrázka: {image_path}')
        if not os.path.exists(image_path):
            print(f'Obrázok neexistuje: {image_path}, preskakujem.')
            continue

        with Image.open(image_path) as img:
            image_dimensions = img.size + (len(img.getbands()),)
        
        # For each annotation create a list with details
        detailed_annotations = [{
            'label': label,
            'bbox': [int(bbox[0] * image_dimensions[0]),  # Xmin
                     int(bbox[1] * image_dimensions[1]),  # Ymin
                     int(bbox[2] * image_dimensions[0]),  # Width
                     int(bbox[3] * image_dimensions[1])]  # Height
        } for bbox, label in zip(annotation['bboxes'], annotation['labels'])]
        
        print(f'Vytváram VOC XML pre obrázok: {image_id}.jpg')
        create_voc_xml(image_id, detailed_annotations, image_dimensions, voc_output_directory)
        print(f'XML súbor úspešne vytvorený a uložený pre: {image_id}.jpg')