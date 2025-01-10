import os
import clip
import cv2
import torch
import yaml
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

# Create processed_videos directory if it doesn't exist
if not os.path.exists('./processed_videos'):
    os.makedirs('./processed_videos')

class Model:
    def __init__(self, settings_path: str = './settings.yaml'):
        with open(settings_path, "r") as file:
            self.settings = yaml.safe_load(file)

        self.device = self.settings['model-settings']['device']
        self.model_name = self.settings['model-settings']['model-name']
        self.threshold = self.settings['model-settings']['prediction-threshold']
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.labels = self.settings['label-settings']['labels']
        self.labels_ = ['a photo of ' + label for label in self.labels]
        self.text_features = self.vectorize_text(self.labels_)
        self.default_label = self.settings['label-settings']['default-label']

    @torch.no_grad()
    def transform_image(self, image: np.ndarray):
        pil_image = Image.fromarray(image).convert('RGB')
        return self.preprocess(pil_image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def vectorize_text(self, text: list):
        tokens = clip.tokenize(text).to(self.device)
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> dict:
        tf_image = self.transform_image(image)
        image_features = self.model.encode_image(tf_image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ self.text_features.T
        values, indices = similarity[0].topk(1)
        label_index = indices[0].cpu().item()
        confidence = abs(values[0].cpu().item())

        label = self.default_label
        if confidence >= self.threshold:
            label = self.labels[label_index]

        return {'label': label, 'confidence': confidence}

def process_video_with_overlay(video_path, model, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    processed_video_path = './processed_videos/processed_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            prediction = model.predict(frame)
            text = f"{prediction['label']} ({prediction['confidence']:.2f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    return processed_video_path

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_extension = file.filename.split('.')[-1].lower()

    if file_extension in ['jpg', 'jpeg', 'png', 'bmp']:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        prediction = model.predict(image)
        return jsonify(prediction)

    elif file_extension in ['mp4', 'avi', 'mov', 'mkv']:
        temp_video_path = './temp_video.' + file_extension
        with open(temp_video_path, 'wb') as temp_file:
            temp_file.write(file.read())

        processed_video_path = process_video_with_overlay(temp_video_path, model)
        os.remove(temp_video_path)
        return jsonify({"processed_video": f"processed_videos/{os.path.basename(processed_video_path)}"})

    else:
        return jsonify({"error": "Unsupported file format"}), 400

@app.route('/processed_videos/<path:filename>', methods=['GET'])
def serve_file(filename):
    return send_from_directory('./processed_videos', filename)

if __name__ == '__main__':
    model = Model(settings_path='./settings.yaml')
    app.run(host='0.0.0.0', port=5000, debug=True)
