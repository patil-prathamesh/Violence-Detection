from flask import Flask,render_template, request, jsonify
import os
import subprocess

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create uploads folder if it doesn't exist

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Get the filename
    filename = file.filename

    # Ensure the file is an image or video
    if filepath.endswith(('.jpg', '.jpeg', '.png', '.mov', '.mp4')):
        command = ['python3', 'run.py', '--image-path', filepath]
    else:
        return jsonify({"error": "Only image files are supported for now"}), 400

    # Run the command using subprocess
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        output = result.stdout  # Capture the standard output
        frames = []
        
        # Parse the output into frames
        for line in output.split("\n"):
            if "Frame" in line and "Predicted label" in line:
                frame_data = line.split("Predicted label:")
                frame_number = frame_data[0].replace('Frame', '').strip()
                label = frame_data[1].strip()
                frames.append({"frame": frame_number, "label": label})

        return jsonify({"message": "Analysis complete", "frames": frames})

    except subprocess.CalledProcessError as e:
        error_message = e.stderr  # Capture the error output
        return jsonify({"error": "Failed to process the file", "details": error_message}), 500

if __name__ == '__main__':
    app.run(debug=True)
