# import argparse

# import cv2

# from model import Model


# def argument_parser():
#     parser = argparse.ArgumentParser(description="Violence detection")
#     parser.add_argument('--image-path', type=str,
#                         default='./data/7.jpg',
#                         help='path to your image')
#     args = parser.parse_args()
#     return args


# if __name__ == '__main__':
#     args = argument_parser()
#     model = Model()
#     image = cv2.imread(args.image_path)
#     label = model.predict(image=image)['label']
#     print('predicted label: ', label)
#     cv2.imshow(label.title(), image)
#     cv2.waitKey(0)

# -----------------------

# import argparse
# import cv2
# from model import Model


# def argument_parser():
#     parser = argparse.ArgumentParser(description="Violence detection")
#     parser.add_argument('--image-path', type=str,
#                         default='./data/7.jpg',
#                         help='Path to your image or video file')
#     args = parser.parse_args()
#     return args


# def process_video(video_path, model):
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         raise ValueError(f"Failed to open video: {video_path}")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break  # End of video

#         label = model.predict(image=frame)['label']
#         print('Predicted label: ', label)
#         cv2.imshow(label.title(), frame)

#         # Break the loop on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == '__main__':
#     args = argument_parser()
#     model = Model()

#     # Check if input is an image or video
#     if args.image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
#         # Process as an image
#         image = cv2.imread(args.image_path)
#         if image is None:
#             raise ValueError(f"Failed to load image: {args.image_path}")
#         label = model.predict(image=image)['label']
#         print('Predicted label: ', label)
#         cv2.imshow(label.title(), image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     else:
#         # Process as a video
#         process_video(args.image_path, model)

# ------------------------

import argparse
import cv2
from model import Model


def argument_parser():
    parser = argparse.ArgumentParser(description="Violence detection")
    parser.add_argument('--image-path', type=str,
                        default='./data/7.jpg',
                        help='Path to your image or video file')
    args = parser.parse_args()
    return args


def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    # Get the video frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)  # Delay in milliseconds between frames

    frame_counter = 0
    skip_frames = 10  # Process every 10th frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Only process every nth frame
        if frame_counter % skip_frames == 0:
            label = model.predict(image=frame)['label']
            print(f'Frame {frame_counter}: Predicted label: {label}')

        frame_counter += 1

        # Display the current frame
        cv2.imshow("Violence Detection", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_image(image_path, model):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Predict the label
    label = model.predict(image=image)['label']
    print('Predicted label: ', label)

    # Display the image
    cv2.imshow(label.title(), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = argument_parser()
    model = Model()

    # Check if input is an image or video
    if args.image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        process_image(args.image_path, model)
    else:
        process_video(args.image_path, model)
