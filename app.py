# import cv2
# import numpy as np
# import streamlit as st
# from PIL import Image


# @st.cache(allow_output_mutation=True)
# def get_predictor_model():
#     from model import Model
#     model = Model()
#     return model


# header = st.container()
# model = get_predictor_model()

# with header:
#     st.title('Hello!')
#     st.text(
#         'Using this app you can classify whether there is fight on a street? or fire? or car crash? or everything is okay?')

# uploaded_file = st.file_uploader("Or choose an image...")
# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert('RGB')
#     image = np.array(image)
#     label_text = model.predict(image=image)['label'].title()
#     st.write(f'Predicted label is: **{label_text}**')
#     st.write('Original Image')
#     if len(image.shape) == 3:
#         cv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     st.image(image)
# -------------------

from flask import Flask, request, render_template, send_file, jsonify
import os
import cv2
from model import Model
import numpy as np

app = Flask(__name__)

# Initialize the model
model = Model()

# Define upload and processed folders
UPLOAD_FOLDER = './uploads'
PROCESSED_FOLDER = './processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_file():
    file = request.files.get('file')

    if not file:
        return "No file uploaded.", 400

    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # Process image
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        prediction = model.predict(image=image)
        label = prediction['label']
        confidence = prediction['confidence']

        return jsonify({
            "type": "image",
            "label": label,
            "confidence": confidence
        })

    elif file.filename.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
        # Process video
        cap = cv2.VideoCapture(file_path)
        frame_counter = 0
        skip_frames = 10  # Process every 10th frame
        label = "Processing..."
        processed_video_path = os.path.join(PROCESSED_FOLDER, f"processed_{file.filename}")
        out = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process every nth frame
            if frame_counter % skip_frames == 0:
                label = model.predict(image=frame)['label']

            # Add label as overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            position = (50, 50)
            font_scale = 1
            color = (0, 0, 255)
            thickness = 2
            cv2.putText(frame, f"Label: {label}", position, font, font_scale, color, thickness)

            # Write the frame to the output video
            if out is None:
                height, width, _ = frame.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(processed_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
            out.write(frame)

            frame_counter += 1

        cap.release()
        if out:
            out.release()

        return jsonify({
            "type": "video",
            "processed_video": processed_video_path
        })

    else:
        return "Unsupported file format. Please upload an image or video.", 400


@app.route('/processed/<filename>')
def serve_processed_file(filename):
    return send_file(os.path.join(PROCESSED_FOLDER, filename))


if __name__ == '__main__':
    app.run(debug=True)


