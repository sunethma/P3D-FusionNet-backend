---
title: P3D_FusionNet_backend
emoji: 🌍
colorFrom: pink
colorTo: purple
sdk: docker
pinned: false
---

# 📦 P3D-FusionNet: Single-View 3D Reconstruction

P3D-FusionNet is my Final Year Project, built on top of the [3D-RETR (BMVC 2021)](https://www.bmvc2021-virtualconference.com/conference/papers/paper_1112.html) framework.
It reconstructs 3D voxel models from single 2D images, enhanced with Transformer-in-Transformer (TNT) feature extraction and Performer-based attention for efficient computation.

🚀 A deployed demo (backend + trained model) is available on Hugging Face Spaces:

👉 [Hugging Face Space - P3D-FusionNet](https://huggingface.co/spaces/Sunethma/P3D_FusionNet_backend/tree/main)

--- 

# 📖 Table of Contents

[Features](Features)

[Project Structure](Project Structure)

[Environment Setup]

[Dataset]

[Training]

[Evaluation]

[Demo]

[Acknowledgements]

[Citation]

---


# ✨ Features

Upload a single 2D image → generate a 3D voxel model.

Enhanced feature extraction using TNT.

Efficient linear attention with Performer.

Trained & tested on ShapeNet (13 categories).

Backend hosted on Hugging Face Spaces.

---
# 📁 Project Structure
```
P3D-FusionNet/
│── config/                  # Config files for model and training
│── src/                     # Source code (model, dataloaders, utils)
│── app.py                   # Backend app entry point (Hugging Face)
│── requirements.txt         # Python dependencies
│── Dockerfile               # Container setup for deployment
│── output.binvox            # Example generated voxel output
│── README.md                # This file

```

---


