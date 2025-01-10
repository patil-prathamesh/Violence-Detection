import argparse
import cv2
import numpy as np
from model import Model


def argument_parser():
    parser = argparse.ArgumentParser(description="Violence detection")
    parser.add_argument('--image-path', type=str,
                        default='./data/7.jpg',
                        help='Path to your image or video file')
    args = parser.parse_args()
    return args


def process_video(video_path, model, output_size=(640, 480)):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    # Get the video frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)  # Delay in milliseconds between frames

    frame_counter = 0
    skip_frames = 10  # Process every 10th frame
    label = "Processing..."  # Default label

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Only process every nth frame
        if frame_counter % skip_frames == 0:
            label = model.predict(image=frame)['label']
            print(f'Frame {frame_counter}: Predicted label: {label}')

        frame_counter += 1

        # Add a white border above the frame
        border_height = 100  # Increased height of the white border
        border = np.full((border_height, frame.shape[1], 3), 255, dtype=np.uint8)  # Create a white border
        frame_with_border = cv2.vconcat([border, frame])  # Concatenate border and frame

        # Overlay the label on the white border
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5  # Increased font scale
        font_thickness = 3  # Increased font thickness
        text_color = (0,0,0)  # Green color
        margin_top = 20  # Adjust this value for more or less margin
        position = (20, 60 + margin_top)  # Adjusted Y-coordinate for top margin
        cv2.putText(frame_with_border, f"Label: {label}", position, font, font_scale, text_color, font_thickness)

        # Resize the frame to the desired output size
        frame_resized = cv2.resize(frame_with_border, output_size)

        # Display the current frame with the label
        cv2.imshow("Violence Detection", frame_resized)

        # Break the loop on 'q' key press
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_image(image_path, model, output_size=(640, 480)):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Predict the label
    label = model.predict(image=image)['label']
    print('Predicted label: ', label)

    # Add a white border above the image
    border_height = 100  # Increased height of the white border
    border = np.full((border_height, image.shape[1], 3), 255, dtype=np.uint8)  # Create a white border
    image_with_border = cv2.vconcat([border, image])  # Concatenate border and image

    # Overlay the label on the white border
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5  # Increased font scale
    font_thickness = 3  # Increased font thickness
    text_color = (0,0,0)  # Blue text in OpenCV (BGR format)
    margin_top = 20  # Adjust this value for more or less margin
    position = (20, 40 + margin_top)  # Adjusted position for the larger text
    cv2.putText(image_with_border, f"Label: {label}", position, font, font_scale, text_color, font_thickness)

    # Resize the image to the desired output size
    image_resized = cv2.resize(image_with_border, output_size)

    # Display the image with the label
    cv2.imshow(label.title(), image_resized)
    
    cv2.waitKey(7000)
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = argument_parser()
    model = Model()

    # Check if input is an image or video
    if args.image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        process_image(args.image_path, model)
    else:
        process_video(args.image_path, model)
