from flask import Flask, request
import torch
from PIL import Image
import numpy as np
from io import BytesIO
import base64

app = Flask(__name__)

# Load quantized model
model = torch.load('savedmodel.pth', map_location='cpu')
model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    pred_text = ""
    img_b64 = ""
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            stream = file.stream
            stream.seek(0)
            original_img = Image.open(stream)
            stream.seek(0)
            img = Image.open(stream).convert('L').resize((64, 64))
            
            arr = np.array(img) / 255.0
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
            
            with torch.no_grad():
                pred = int(model(tensor).argmax(1).item())
            
            pred_text = f"<h2>Predicted Person ID: {pred}</h2>"
            
            buffered = BytesIO()
            original_img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()

            img_tag = f'<img src="data:image/png;base64,{img_b64}" style="max-width:600px;"><br>(original uploaded image)'

            return f"<h1>Olivetti Faces Classifier</h1>{pred_text}{img_tag}<br><a href='/'>Try another</a>"

    return '''
    <h1>Upload a face image (will be resized to 64x64 grayscale)</h1>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <input type="submit" value="Predict">
    </form>
    '''
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)