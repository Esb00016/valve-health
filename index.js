const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const app = express();
const port = process.env.PORT || 3000;

// Crear el directorio `uploads` si no existe
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)){
    fs.mkdirSync(uploadsDir, { recursive: true });
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
    // Procesamiento del archivo y predicción
    try {
      const audioFilePath = req.file.path;
      const prediction = await makePrediction(audioFilePath);
      res.json({ prediction });
    } catch (error) {
      res.status(500).send(error.message);
    }
  } else {
    res.status(400).send('File upload failed.');
  }
});

async function makePrediction(filePath) {
  // Cargar el modelo
  const model = await tf.loadLayersModel('file://model_tfjs/model.json');
  
  // Aquí deberías implementar la función para extraer el vector de características del archivo de audio
  const features = await extractFeaturesFromAudio(filePath);
  
  const inputTensor = tf.tensor2d([features], [1, 1200]);
  const prediction = model.predict(inputTensor);
  const predictionArray = await prediction.array();
  return predictionArray;
}

async function extractFeaturesFromAudio(filePath) {
  // Implementa la lógica para procesar el archivo .wav y extraer un vector de 1200 características
  // Este es solo un ejemplo y necesitarás adaptar esta función a tus necesidades específicas
  const audioBuffer = fs.readFileSync(filePath);
  const audioData = tf.node.decodeWav(audioBuffer);
  const samples = audioData.audio.subarray(0, 1200);
  const normalizedSamples = samples.map(s => s / 32768);
  return normalizedSamples;
}

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
