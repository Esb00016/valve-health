const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const wav = require('node-wav');
const SoxCommand = require('sox-audio');
const app = express();
const port = process.env.PORT || 3000;

// Crear el directorio `uploads` si no existe
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
    console.log('Uploads directory created');
} else {
    console.log('Uploads directory already exists');
}

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, uploadsDir);
  },
  filename: function (req, file, cb) {
    cb(null, file.fieldname + '-' + Date.now() + path.extname(file.originalname));
  }
});

const upload = multer({ storage: storage });

app.use(express.static('public'));

app.post('/upload', upload.single('audiofile'), async (req, res) => {
  if (req.file) {
    console.log(`File uploaded to: ${req.file.path}`);
    // Procesamiento del archivo y predicción
    try {
      const audioFilePath = req.file.path;
      const prediction = await makePrediction(audioFilePath);
      res.json({ prediction });
    } catch (error) {
      console.error('Error processing file:', error);
      res.status(500).send('Error processing file: ' + error.message);
    }
  } else {
    res.status(400).send('File upload failed.');
  }
});

async function makePrediction(filePath) {
  try {
    // Cargar el modelo
    const modelPath = path.join(__dirname, 'model_tfjs', 'model.json');
    const model = await tf.loadLayersModel('file://' + modelPath);
    console.log('Model loaded successfully');

    // Extraer características del archivo de audio
    const features = await extractFeaturesFromAudio(filePath);
    console.log('Features extracted:', features);

    const inputTensor = tf.tensor2d([features], [1, 1200]);
    const prediction = model.predict(inputTensor);
    const predictionArray = await prediction.array();
    return predictionArray;
  } catch (error) {
    console.error('Error in makePrediction:', error);
    throw error;
  }
}

async function extractFeaturesFromAudio(filePath) {
  try {
    // Leer el archivo de audio
    const buffer = fs.readFileSync(filePath);
    const result = wav.decode(buffer);

    const originalSampleRate = result.sampleRate;
    const originalSamples = result.channelData[0];
    const targetSampleRate = 1000;

    // Remuestrear a 1 kHz usando Sox
    const resampledFilePath = filePath.replace('.wav', '_resampled.wav');
    await new Promise((resolve, reject) => {
      SoxCommand()
        .input(filePath)
        .output(resampledFilePath)
        .outputSampleRate(targetSampleRate)
        .run()
        .on('end', resolve)
        .on('error', reject);
    });

    const resampledBuffer = fs.readFileSync(resampledFilePath);
    const resampledResult = wav.decode(resampledBuffer);
    const resampledSamples = resampledResult.channelData[0];

    // Asegurarse de que el archivo tenga 1200 muestras
    let featureVector = resampledSamples.slice(0, 1200);
    if (featureVector.length < 1200) {
      featureVector = featureVector.concat(new Array(1200 - featureVector.length).fill(0));
    }

    // Normalizar las muestras entre 0 y 1
    const maxVal = Math.max(...featureVector);
    const minVal = Math.min(...featureVector);
    const normalizedSamples = featureVector.map(s => (s - minVal) / (maxVal - minVal));

    return normalizedSamples;
  } catch (error) {
    console.error('Error extracting features:', error);
    throw error;
  }
}

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
