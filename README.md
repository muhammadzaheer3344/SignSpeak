# SignSpeak - Real-Time Sign Language Recognition

A computer vision project that classifies American Sign Language (ASL) hand signs using a custom CNN.

---

## Project Overview

SignSpeak recognizes **29 ASL classes** (A-Z + del + nothing + space) from hand sign images with **96.59% validation accuracy**.

### Key Features

- Custom CNN architecture (2.4M parameters)
- Real-time image classification
- Confidence scores + Top-5 predictions
- Streamlit web interface
# SignSpeak

Streamlit image classifier for 29 American Sign Language classes: A-Z, `del`,
`nothing`, and `space`.

**Live app:** https://signspeak-9vjpjaaeb58f3i2vitdtqi.streamlit.app

## Run Locally

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Upload a JPG, JPEG, or PNG image of a hand sign. The app displays the predicted
class, confidence, and top five predictions.

## Deployment Files

```text
app.py
requirements.txt
models/
	baseline_cnn_final.keras
```

Training data, notebooks, checkpoints, logs, and local archives are excluded by
`.gitignore` and remain local.

## Model

The included custom CNN accepts RGB images resized to `64x64` and predicts 29
classes. The reported validation accuracy is 96.59%.

## Streamlit Community Cloud

1. Create or select the GitHub repository.
2. Choose `app.py` as the main file.
3. Deploy with the default Python environment.

The deployment uses the TensorFlow release candidate that provides a wheel for
Streamlit Cloud's Python 3.14 runtime. The model is loaded from
`models/baseline_cnn_final.keras`.

MIT
