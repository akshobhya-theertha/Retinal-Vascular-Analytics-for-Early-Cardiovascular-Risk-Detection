"""
Retinexia - Flask app. Open in Chrome: http://localhost:5000/
"""
import base64
import io
import json
from pathlib import Path

from flask import Flask, request, jsonify
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DR_CLASSES = ["Normal", "Mild", "Moderate", "Severe", "Proliferative"]
CVD_CLASSES = ["Low", "Moderate", "High"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

MODEL_PATH = MODELS_DIR / "retinexia_best.keras"
FALLBACK_MODEL = MODELS_DIR / "retinexia_final.keras"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"
model = None
dr_classes = DR_CLASSES
cvd_classes = CVD_CLASSES


def load_model():
    global model, dr_classes, cvd_classes
    if model is not None:
        return True
    try:
        from tensorflow import keras
        if MODEL_PATH.exists():
            model = keras.models.load_model(str(MODEL_PATH))
        elif FALLBACK_MODEL.exists():
            model = keras.models.load_model(str(FALLBACK_MODEL))
        else:
            model = None
    except Exception:
        model = None
    if CLASS_INDICES_PATH.exists():
        try:
            with open(CLASS_INDICES_PATH, encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d.get("dr_classes"), list) and len(d["dr_classes"]) >= 5:
                dr_classes = d["dr_classes"]
            if isinstance(d.get("cvd_classes"), list) and len(d["cvd_classes"]) >= 3:
                cvd_classes = d["cvd_classes"]
        except Exception:
            pass
    return model is not None


# Inline HTML - no template file needed
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Retinexia - Retinal Vascular Analytics</title>
<style>
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; background: #0a1628; color: #e8eef4; min-height: 100vh; }
.wrap { max-width: 1200px; margin: 0 auto; padding: 24px; }
header { text-align: center; padding: 20px 0; border-bottom: 1px solid #1e3a5f; }
h1 { font-size: 1.75rem; font-weight: 600; color: #7eb8da; margin: 0; letter-spacing: 0.02em; }
.sub { color: #6b8ba4; font-size: 0.9rem; margin-top: 6px; }
.panels { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 24px; }
@media (max-width: 900px) { .panels { grid-template-columns: 1fr; } }
.panel { background: #122236; border-radius: 12px; border: 1px solid #1e3a5f; padding: 24px; }
.panel h2 { font-size: 1rem; color: #7eb8da; margin: 0 0 16px 0; font-weight: 600; }
.upload { border: 2px dashed #2d5a87; padding: 32px; text-align: center; cursor: pointer; background: #0d1f33; border-radius: 10px; color: #8ba3b8; transition: background 0.2s, border-color 0.2s; }
.upload:hover { background: #152a42; border-color: #3d7ab5; color: #b8d4e8; }
input[type="file"] { display: none; }
#previewWrap { margin-top: 12px; text-align: center; }
#preview { max-width: 100%; max-height: 220px; border-radius: 8px; border: 1px solid #1e3a5f; }
.btn { background: #2563eb; color: #fff; border: none; padding: 12px 24px; font-size: 15px; border-radius: 8px; cursor: pointer; margin-top: 12px; font-weight: 500; }
.btn:hover { background: #3b82f6; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.results-panel .stat { display: flex; align-items: center; justify-content: space-between; padding: 12px 16px; background: #0d1f33; border-radius: 8px; margin-bottom: 10px; }
.results-panel .stat label { color: #8ba3b8; font-size: 0.9rem; }
.results-panel .stat strong { color: #7eb8da; font-size: 1.05rem; }
.heatmap-block { margin-top: 20px; }
.heatmap-block h3 { font-size: 0.9rem; color: #6b8ba4; margin: 0 0 8px 0; font-weight: 500; }
.heatmap-img { width: 100%; max-width: 320px; height: auto; border-radius: 8px; border: 1px solid #1e3a5f; display: block; }
#demoMsg { background: #3d3520; color: #e8d68a; padding: 12px; border-radius: 8px; font-size: 0.9rem; margin-bottom: 16px; display: none; }
#errMsg { color: #f87171; margin-top: 12px; display: none; }
.link-bar { text-align: center; margin-top: 16px; }
.link-bar a { color: #7eb8da; text-decoration: none; }
.link-bar a:hover { text-decoration: underline; }
</style>
</head>
<body>
<div class="wrap">
<header>
<h1>Retinexia</h1>
<p class="sub">Retinal Vascular Analytics — DR stage & CVD risk</p>
<div class="link-bar"><a href="http://localhost:5000/">http://localhost:5000/</a></div>
</header>

<div class="panels">
<div class="panel">
<h2>Upload fundus image</h2>
<div class="upload" id="uploadZone">Click or drop retinal image here</div>
<input type="file" id="fileInput" accept="image/*">
<div id="previewWrap" style="display:none;"><img id="preview" alt="Preview"></div>
<button type="button" class="btn" id="btnAnalyze" style="display:none;">Analyze</button>
</div>

<div class="panel results-panel" id="resultsBox" style="display:none;">
<h2>Results</h2>
<p id="demoMsg"></p>
<div class="stat"><label>DR stage</label><strong id="drStage">-</strong> <span id="drConf"></span></div>
<div class="stat"><label>CVD risk</label><strong id="cvdRisk">-</strong> <span id="cvdConf"></span></div>
<div class="heatmap-block"><h3>DR attention map</h3><img id="imgDr" class="heatmap-img" style="display:none;" alt="DR heatmap"></div>
<div class="heatmap-block"><h3>CVD attention map</h3><img id="imgCvd" class="heatmap-img" style="display:none;" alt="CVD heatmap"></div>
</div>
</div>
<p id="errMsg"></p>
</div>

<script>
var zone = document.getElementById('uploadZone');
var fileInput = document.getElementById('fileInput');
var preview = document.getElementById('preview');
var previewWrap = document.getElementById('previewWrap');
var btnAnalyze = document.getElementById('btnAnalyze');
var resultsBox = document.getElementById('resultsBox');
var drStage = document.getElementById('drStage');
var drConf = document.getElementById('drConf');
var cvdRisk = document.getElementById('cvdRisk');
var cvdConf = document.getElementById('cvdConf');
var imgDr = document.getElementById('imgDr');
var imgCvd = document.getElementById('imgCvd');
var demoMsg = document.getElementById('demoMsg');
var errMsg = document.getElementById('errMsg');

zone.onclick = function() { fileInput.click(); };
zone.ondragover = function(e) { e.preventDefault(); zone.style.background = '#90caf9'; };
zone.ondragleave = function() { zone.style.background = '#bbdefb'; };
zone.ondrop = function(e) {
  e.preventDefault();
  zone.style.background = '#bbdefb';
  if (e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; onSelect(); }
};
fileInput.onchange = function() { onSelect(); };

function onSelect() {
  if (!fileInput.files[0]) return;
  preview.src = URL.createObjectURL(fileInput.files[0]);
  previewWrap.style.display = 'block';
  btnAnalyze.style.display = 'block';
  resultsBox.style.display = 'none';
  errMsg.style.display = 'none';
}

btnAnalyze.onclick = function() {
  var file = fileInput.files[0];
  if (!file) return;
  btnAnalyze.disabled = true;
  btnAnalyze.textContent = 'Analyzing...';
  errMsg.style.display = 'none';
  var fd = new FormData();
  fd.append('file', file);
  fetch('/predict', { method: 'POST', body: fd })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      drStage.textContent = data.dr_stage || '-';
      drConf.textContent = (data.dr_confidence != null ? (data.dr_confidence * 100).toFixed(1) + '%' : '');
      cvdRisk.textContent = data.cvd_risk || '-';
      cvdConf.textContent = (data.cvd_confidence != null ? (data.cvd_confidence * 100).toFixed(1) + '%' : '');
      if (data.heatmap_dr_base64) {
        imgDr.src = 'data:image/png;base64,' + data.heatmap_dr_base64;
        imgDr.style.display = 'block';
      } else { imgDr.style.display = 'none'; }
      if (data.heatmap_cvd_base64) {
        imgCvd.src = 'data:image/png;base64,' + data.heatmap_cvd_base64;
        imgCvd.style.display = 'block';
      } else { imgCvd.style.display = 'none'; }
      demoMsg.style.display = data.demo ? 'block' : 'none';
      if (data.message) demoMsg.textContent = data.message;
      resultsBox.style.display = 'block';
      resultsBox.scrollIntoView();
    })
    .catch(function(e) {
      errMsg.textContent = e.message || 'Request failed';
      errMsg.style.display = 'block';
    })
    .finally(function() {
      btnAnalyze.disabled = false;
      btnAnalyze.textContent = 'Analyze';
    });
};
</script>
</body>
</html>"""


@app.route("/")
def index():
    return HTML_PAGE


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files and "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files.get("file") or request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if model is None:
        load_model()
    if model is None:
        try:
            Image.open(io.BytesIO(file.read())).convert("RGB")
        except Exception as e:
            return jsonify({"error": "Invalid image: " + str(e)}), 400
        return jsonify({
            "dr_stage": "Normal", "dr_confidence": 0.85,
            "cvd_risk": "Low", "cvd_confidence": 0.82,
            "heatmap_dr_base64": None, "heatmap_cvd_base64": None,
            "demo": True, "message": "Demo mode. Run train.py for real predictions.",
        })

    try:
        import numpy as np
        from preprocess import preprocess_single_image
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": "Invalid image: " + str(e)}), 400
    arr = np.array(img)
    preprocessed = preprocess_single_image(arr)
    dr_out, cvd_out = model.predict(preprocessed, verbose=0)
    dr_proba, cvd_proba = dr_out[0], cvd_out[0]
    dr_idx = int(np.argmax(dr_proba))
    cvd_idx = int(np.argmax(cvd_proba))
    dr_stage = dr_classes[dr_idx] if dr_idx < len(dr_classes) else "Class_" + str(dr_idx)
    cvd_risk = cvd_classes[cvd_idx] if cvd_idx < len(cvd_classes) else "Level_" + str(cvd_idx)
    heatmap_dr_b64 = None
    heatmap_cvd_b64 = None
    try:
        from gradcam import make_gradcam_model, compute_heatmap, overlay_heatmap_on_image
        gmodel, _ = make_gradcam_model(model)
        h = compute_heatmap(gmodel, preprocessed, output_index=0)
        ov = overlay_heatmap_on_image(preprocessed, h, alpha=0.5)
        buf = io.BytesIO()
        Image.fromarray(ov).save(buf, format="PNG")
        heatmap_dr_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        h = compute_heatmap(gmodel, preprocessed, output_index=1)
        ov = overlay_heatmap_on_image(preprocessed, h, alpha=0.5)
        buf = io.BytesIO()
        Image.fromarray(ov).save(buf, format="PNG")
        heatmap_cvd_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        pass
    return jsonify({
        "dr_stage": dr_stage, "dr_confidence": float(dr_proba[dr_idx]),
        "cvd_risk": cvd_risk, "cvd_confidence": float(cvd_proba[cvd_idx]),
        "heatmap_dr_base64": heatmap_dr_b64, "heatmap_cvd_base64": heatmap_cvd_b64,
    })


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    url = "http://localhost:" + str(port) + "/"
    print("\n" + "=" * 50)
    print("  RETINEXIA - Copy this link into Chrome:")
    print("  " + url)
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)
