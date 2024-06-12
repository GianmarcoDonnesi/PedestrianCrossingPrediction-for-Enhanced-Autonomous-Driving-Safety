import os
import xml.etree.ElementTree as ET
import numpy as np
import pickle

def preprocess_annotations(annotations_base_dir, cache_dir, video_names):
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define paths to various annotation directories
    annotations_dir = os.path.join(annotations_base_dir, 'annotations')
    annotations_dirs = {
        'attributes': os.path.join(annotations_base_dir, 'annotations_attributes'),
        'appearance': os.path.join(annotations_base_dir, 'annotations_appearance'),
        'traffic': os.path.join(annotations_base_dir, 'annotations_traffic'),
        'vehicle': os.path.join(annotations_base_dir, 'annotations_vehicle')
    }
    
# Process each video
    for video in video_names:
        video_id = video.split('_')[1].split('.')[0]
        
        # Define paths to annotation files for the current video
        annotations_paths = {
            'annotations': os.path.join(annotations_dir, f"video_{video_id}.xml"),
            'attributes': os.path.join(annotations_dirs['attributes'], f"video_{video_id}_attributes.xml"),
            'appearance': os.path.join(annotations_dirs['appearance'], f"video_{video_id}_appearance.xml"),
            'traffic': os.path.join(annotations_dirs['traffic'], f"video_{video_id}_traffic.xml"),
            'vehicle': os.path.join(annotations_dirs['vehicle'], f"video_{video_id}_vehicle.xml")
        }
        
        # Parse the XML files if they exist
        annotations = {key: ET.parse(path).getroot() for key, path in annotations_paths.items() if os.path.exists(path)}

        preprocessed_data = []
        
        # Extract data from the main annotations file
        for track in annotations['annotations'].findall('.//track'):
            for box in track.findall('.//box'):
                frame_id = int(box.get('frame'))
                label = 1 if 'behavior' in track.attrib and track.attrib['behavior'] == 'crossing' else 0
                traffic_info = get_traffic_info(annotations.get('traffic', None), frame_id)
                vehicle_info = get_vehicle_info(annotations.get('vehicle', None), frame_id)
                appearance_info = get_appearance_info(annotations.get('appearance', None), frame_id)
                attributes_info = get_attributes_info(annotations.get('attributes', None), frame_id)
                
                # Append preprocessed data
                preprocessed_data.append((frame_id, label, traffic_info, vehicle_info, appearance_info, attributes_info))
                
        # Save the preprocessed data to a pickle file
        with open(os.path.join(cache_dir, f"video_{video_id}.pkl"), 'wb') as f:
            pickle.dump(preprocessed_data, f)

def get_traffic_info(root, frame_id):
     # Default traffic information
    traffic_info = {'ped_crossing': 0, 'ped_sign': 0, 'stop_sign': 0, 'traffic_light': 0}
    if root is not None:
        for frame in root.findall('.//frame'):
            if int(frame.get('id')) == frame_id:
                # Update traffic information if the frame matches the frame_id
                traffic_info = {
                    'ped_crossing': int(frame.get('ped_crossing')),
                    'ped_sign': int(frame.get('ped_sign')),
                    'stop_sign': int(frame.get('stop_sign')),
                    'traffic_light': 1 if frame.get('traffic_light') != 'n/a' else 0
                }
    return traffic_info

def get_vehicle_info(root, frame_id):
    # Default vehicle information
    vehicle_info = {'action': 0}
    if root is not None:
        for frame in root.findall('.//frame'):
            if int(frame.get('id')) == frame_id:
                action = frame.get('action')
                # Update vehicle information based on action
                vehicle_info = {
                    'action': 1 if action == 'moving_slow' else 2 if action == 'decelerating' else 3 if action == 'stopped' else 4 if action is not None and action == 'accelerating' else 0
                }
    return vehicle_info

def get_appearance_info(root, frame_id):
    # Default appearance information
    appearance_info = {'pose': 0, 'clothing': 0, 'objects': 0}
    if root is not None:
        for track in root.findall('.//track'):
            for box in track.findall('.//box'):
                if int(box.get('frame')) == frame_id:
                    # Update appearance information if the frame matches the frame_id
                    appearance_info = {
                        'pose': float(box.find('pose').text) if box.find('pose') is not None else 0,
                        'clothing': float(box.find('clothing').text) if box.find('clothing') is not None else 0,
                        'objects': float(box.find('objects').text) if box.find('objects') is not None else 0
                    }
    return appearance_info

def get_attributes_info(root, frame_id):
    # Default attributes information
    attributes_info = {'age': 0, 'gender': 0, 'crossing_point': 0}
    if root is not None:
        for track in root.findall('.//track'):
            for box in track.findall('.//box'):
                if int(box.get('frame')) == frame_id:
                    # Update attributes information if the frame matches the frame_id
                    attributes_info = {
                        'age': float(box.find('age').text) if box.find('age') is not None else 0,
                        'gender': 1 if box.find('gender') is not None and box.find('gender').text == 'male' else 0 if box.find('gender') is not None and box.find('gender').text == 'female' else 0,
                        'crossing_point': float(box.find('crossing_point').text) if box.find('crossing_point') is not None else 0
                    }
    return attributes_info

annotations_base_dir = './JAAD_dataset/'
# Stores preprocessed data for each video in a pickle file within the cache directory.
cache_dir = './JAAD_dataset/cache'
video_names = sorted(os.listdir(annotations_base_dir + 'annotations'))[:102]
preprocess_annotations(annotations_base_dir, cache_dir, video_names)

print("Annotations preprocessed")