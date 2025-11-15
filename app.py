# app.py
from flask import Flask, request, render_template_string
from PIL import Image
import numpy as np
import joblib
import io
import base64

app = Flask(__name__)
model = joblib.load('savedmodel.pth')

HTML = '''
<h1>Olivetti Faces Classifier</h1>
<form method="post" enctype="multipart/form-data">
    <input type="file" name="file" accept="image/*">
    <input type="submit" value="Predict">
</form>
{% if pred %}
<h2>Predicted Person ID: {{ pred }}</h2>
<img src="data:image/png;base64,{{ img }}" style="max-width:400px;">
{% endif %}
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    pred = None
    img = None
    if request.method == 'POST':
        file = request.files['file']
        original = Image.open(file.stream)
        img_pil = original.convert('L').resize((64, 64))
        arr = np.array(img_pil).flatten() / 255.0
        arr = arr.reshape(1, -1)
        pred = int(model.predict(arr)[0])

        buffered = io.BytesIO()
        original.save(buffered, format="PNG")
        img = base64.b64encode(buffered.getvalue()).decode()

    return render_template_string(HTML, pred=pred, img=img)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)