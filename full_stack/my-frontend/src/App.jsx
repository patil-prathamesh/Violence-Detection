import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [processedVideo, setProcessedVideo] = useState(null);

  const handleFileChange = (event) => {
    const uploadedFile = event.target.files[0];
    setFile(uploadedFile);
    setPrediction(null);
    setProcessedVideo(null);
  };

  const handleUpload = async () => {
    if (!file) {
      alert('Please upload a file first!');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      if (response.data.processed_video) {
        setProcessedVideo('http://localhost:5000/' + response.data.processed_video);
      } else {
        setPrediction(response.data);
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('An error occurred while processing the file.');
    }
  };

  return (
    <div style={{ padding: '20px', textAlign: 'center' }}>
      <h1>Image and Video Prediction</h1>
      <input type="file" onChange={handleFileChange} accept="image/*,video/*" />
      <button onClick={handleUpload} style={{ marginLeft: '10px' }}>Upload and Predict</button>

      {file && (
        <div style={{ marginTop: '20px' }}>
          <h3>Uploaded File: {file.name}</h3>
          <p>{file.type.startsWith('video/') ? 'Video file detected' : 'Image file detected'}</p>
        </div>
      )}

      {prediction && (
        <div style={{ marginTop: '20px' }}>
          <h3>Prediction:</h3>
          <p>Label: {prediction.label}</p>
          <p>Confidence: {prediction.confidence}</p>
        </div>
      )}

      {processedVideo && (
        <div style={{ marginTop: '20px' }}>
          <h3>Processed Video:</h3>
          <video src={processedVideo} controls style={{ maxWidth: '500px', maxHeight: '500px' }} />
        </div>
      )}
    </div>
  );
}

export default App;
