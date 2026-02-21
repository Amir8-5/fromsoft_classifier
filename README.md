---
title: FromSoftware Classifier
emoji: 🗡️
colorFrom: gray
colorTo: yellow
sdk: docker
pinned: false
---

# FromSoftware Image Classifier

A deep learning API that classifies screenshots from FromSoftware games (Bloodborne, Elden Ring, Dark Souls, etc.) using a fine-tuned ResNet50 model.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/ready` | GET | Readiness check (model loaded) |
| `/api/v1/predict` | POST | Classify an image |

## Usage

POST a JPEG or PNG screenshot (≤ 5 MB) to `/api/v1/predict`:

```bash
curl -X POST https://Amir8-5-fromsoft-classifier.hf.space/api/v1/predict \
  -F "file=@screenshot.jpg"
```
