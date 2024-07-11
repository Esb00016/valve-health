document.getElementById('uploadForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    const fileInput = document.getElementById('audiofile');
    const file = fileInput.files[0];
    
    const formData = new FormData();
    formData.append('audiofile', file);
    
    const response = await fetch('/upload', {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    displayResult(result.prediction);
  });
  
  function displayResult(prediction) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = `<p>Predicción: ${JSON.stringify(prediction)}</p>`;
  }
  