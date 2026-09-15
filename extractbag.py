import argparse
import os
import csv
import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

# ROS encoding -> (channels, cv2 color-conversion to BGR-for-imwrite or None)
_RAW_ENCODINGS = {
    'rgb8': (3, cv2.COLOR_RGB2BGR),
    'bgr8': (3, None),
    'rgba8': (4, cv2.COLOR_RGBA2BGR),
    'bgra8': (4, cv2.COLOR_BGRA2BGR),
    'mono8': (1, None),
}


def decode_image(msg, msgtype):
    """Decode either a CompressedImage (jpeg/png bytes) or a raw Image message."""
    if 'CompressedImage' in msgtype:
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # Raw sensor_msgs/Image: reshape using step (row stride), since step can
    # include padding beyond width * channels.
    channels, conversion = _RAW_ENCODINGS.get(msg.encoding, (3, None))
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    image = raw[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
    if conversion is not None:
        image = cv2.cvtColor(image, conversion)
    return image


def process_bag(file_path, image_topic, cmd_topic):
    # Ensure the file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    # Setup output directories and files
    output_dir = os.path.dirname(os.path.abspath(file_path))
    images_dir = os.path.join(output_dir, 'images')
    csv_file_path = os.path.join(output_dir, 'commands.csv')

    os.makedirs(images_dir, exist_ok=True)

    # Initialize ROS1 Type Store
    typestore = get_typestore(Stores.ROS1_NOETIC)

    print(f"Opening file: {file_path}")
    print(f"Images will be saved to: {images_dir}")
    print(f"Commands will be saved to: {csv_file_path}")

    with Reader(file_path) as reader:
        available_topics = {c.topic for c in reader.connections}
        for topic in (image_topic, cmd_topic):
            if topic not in available_topics:
                print(f"Error: topic '{topic}' not found in this bag.")
                print("Available topics:")
                for t in sorted(available_topics):
                    print(f"  {t}")
                return

        image_count = 0
        command_count = 0

        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['timestamp', 'linear_v', 'angular_w'])

            for connection, timestamp, rawdata in reader.messages():

                # 1. Extract and save images
                if connection.topic == image_topic:
                    msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                    image = decode_image(msg, connection.msgtype)
                    if image is None:
                        print(f"Warning: failed to decode image at timestamp {timestamp}")
                        continue

                    image_filename = os.path.join(images_dir, f'frame_{timestamp}.jpg')
                    cv2.imwrite(image_filename, image)
                    image_count += 1

                # 2. Extract and save velocity commands
                elif connection.topic == cmd_topic:
                    msg = typestore.deserialize_ros1(rawdata, connection.msgtype)

                    linear_v = msg.linear.x
                    angular_w = msg.angular.z

                    csv_writer.writerow([timestamp, linear_v, angular_w])
                    command_count += 1

    print(f"Extraction complete! Saved {image_count} images and {command_count} commands.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images and cmd_vel from a SCAND bag file.")
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the .bag file"
    )
    parser.add_argument(
        "--image-topic",
        type=str,
        default="/image_raw/compressed",
        help="Topic to extract images from (default: /image_raw/compressed). "
             "Other options in Spot SCAND bags: /spot/camera/{frontleft,frontright,back,left,right}/image/compressed",
    )
    parser.add_argument(
        "--cmd-topic",
        type=str,
        default="/navigation/cmd_vel",
        help="Topic to extract velocity commands from (default: /navigation/cmd_vel)",
    )

    args = parser.parse_args()
    process_bag(args.file_path, args.image_topic, args.cmd_topic)
