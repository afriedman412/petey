"""
Web interface for PAR decision extraction.
Drop a PDF, get structured JSON back.
"""
import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse

from par_rag_extract import async_process_file

app = FastAPI()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PAR Extractor</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 700px;
         margin: 2rem auto; padding: 0 1rem; }
  .dropzone {
    border: 2px dashed #adb5bd; border-radius: 8px; padding: 3rem;
    text-align: center; color: #666; cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
  }
  .dropzone.over { border-color: #228be6; background: #e7f5ff; }
  .dropzone input { display: none; }
  #status { margin: 1rem 0; }
  pre { background: #f1f3f5; padding: 1rem; border-radius: 6px;
        overflow-x: auto; font-size: 0.85rem; }
</style>
</head>
<body>
<h1>PAR Decision Extractor</h1>

<div class="dropzone" id="dropzone">
  <p>Drop a PDF here, or click to select</p>
  <input type="file" id="fileInput" accept=".pdf">
</div>
<div id="status"></div>
<pre id="output" style="display:none"></pre>

<script>
const dz = document.getElementById('dropzone');
const fi = document.getElementById('fileInput');
const st = document.getElementById('status');
const out = document.getElementById('output');

dz.addEventListener('click', () => fi.click());
dz.addEventListener('dragover', e => {
  e.preventDefault(); dz.classList.add('over');
});
dz.addEventListener('dragleave', () => dz.classList.remove('over'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('over');
  const f = [...e.dataTransfer.files].find(
    f => f.name.toLowerCase().endsWith('.pdf'));
  if (f) upload(f);
});
fi.addEventListener('change', e => { if (e.target.files[0]) upload(e.target.files[0]); });

async function upload(file) {
  st.textContent = 'Processing ' + file.name + '...';
  out.style.display = 'none';
  const form = new FormData();
  form.append('file', file);
  const resp = await fetch('/extract', { method: 'POST', body: form });
  const data = await resp.json();
  out.textContent = JSON.stringify(data, null, 2);
  out.style.display = 'block';
  st.textContent = 'Done.';
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/extract")
async def extract(file: UploadFile):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = await async_process_file(tmp_path)
        result["_source_file"] = file.filename
    except Exception as e:
        result = {"_source_file": file.filename, "_error": str(e)}
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return result
