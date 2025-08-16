---
title: P3D_FusionNet_backend
emoji: 🌍
colorFrom: pink
colorTo: purple
sdk: docker
pinned: false
---

# 📦 P3D-FusionNet: Single-View 3D Reconstruction

## 📘 Project Overview: P3D-FusionNet

P3D-FusionNet is my Final Year Project, built on top of the 3D-RETR [(BMVC 2021)](https://www.bmvc2021-virtualconference.com/conference/papers/paper_1112.html).
It reconstructs 3D voxel models from single 2D images, enhanced with Transformer-in-Transformer (TNT) feature extraction and Performer-based scalable attention for efficient computation.

## 🔍 Research Motivation & Gaps

Single-view 3D reconstruction remains one of the most challenging problems in computer vision because a single image provides limited spatial and depth information.
Existing approaches have the following gaps:

CNN-based methods → struggle with static feature extraction and often lose fine structural details.

Transformer-based methods → powerful but computationally expensive, not optimized for efficiency in real-time or memory-sensitive scenarios.

Feature extraction → current methods often fail to capture contextual and hierarchical spatial relationships, leading to inaccurate reconstructions for complex shapes.

## 🚀 Our Contribution

P3D-FusionNet addresses these issues with a hybrid architecture that improves both accuracy and efficiency:

TNT-based hierarchical feature extraction → preserves contextual and spatial details from 2D images.

Performer attention → reduces computational complexity while scaling effectively to larger inputs.

Voxel-based output generation → provides memory-efficient and high-quality 3D representations.

## 📌 Significance

Produces more detailed 3D reconstructions from a single image.

Optimized for accuracy + speed, making it suitable for real-time applications and limited-resource environments.

Serves as a research contribution in advancing hybrid neural architectures for single-view 3D reconstruction.

🚀 A deployed demo (backend + trained model) is available on Hugging Face Spaces:

👉 [Hugging Face Space - P3D-FusionNet](https://huggingface.co/spaces/Sunethma/P3D_FusionNet_backend/tree/main)

--- 

## 📖 Table of Contents

- [Features](https://github.com/sunethma/P3D-FusionNet-backend/blob/main/README.md#-features)  

[Project Structure](Project Structure)

[Environment Setup]

[Dataset]

[Training]

[Evaluation]

[Demo]

[Acknowledgements]

[Citation]

---


## ✨ Features

Upload a single 2D image → generate a 3D voxel model.

Enhanced feature extraction using TNT.

Efficient linear attention with Performer.

Trained & tested on ShapeNet (13 categories).

Backend hosted on Hugging Face Spaces.

---
## 📁 Project Structure
```
├── config/
├── data/
├── src/
├── Dockerfile
├── README.md
├── app.py
├── eval.py
├── output.binvox
├── requirements.txt
└── train.py

```

---
## ⚙️ Environment Setup

Clone the repo:

```bash
git clone https://github.com/sunethma/P3D-FusionNet-backend.git
cd P3D-FusionNet-backend

```

Install dependencies:
```bash

pip install -r requirements.txt
```

Alternatively, create environment in Google Colab:
```bash

from google.colab import drive
drive.mount('/content/drive')

# Copy repo between Drive and Colab
import shutil
shutil.copytree('folder location in your google drive', '/content/P3D-FusionNet')

# Install dependencies
!pip install torch torchvision pyyaml pytorch-lightning mlflow gitpython performer-pytorch transformers timm pillow

```

## 📊 Dataset

We use the ShapeNet dataset:

Download Rendered Images and Voxelization (32).

Extract them into the following paths:

```bash
!mkdir ShapeNet/
!wget http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
!tar -xzf ShapeNetRendering.tgz -C ShapeNet/
!wget http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz
!tar -xzf ShapeNetVox32.tgz -C ShapeNet/

SHAPENET_VOX = '/content/ShapeNet/ShapeNetVox32'
SHAPENET_IMAGES = '/content/ShapeNet/ShapeNetRendering'

```

## 🏋️ Training

Example training command (Colab):

```bash

cd /content/P3D-FusionNet
python train.py \
    --model image2voxel \
    --transformer_config config/3d-retr-b.yaml \
    --annot_path data/ShapeNet.json \
    --model_path $SHAPENET_VOX \
    --image_path $SHAPENET_IMAGES \
    --gpus 1 \
    --precision 16 \
    --deterministic \
    --train_batch_size 16 \
    --val_batch_size 16 \
    --num_workers 4 \
    --check_val_every_n_epoch 1 \
    --accumulate_grad_batches 1 \
    --view_num 1 \
    --sample_batch_num 0 \
    --loss_type dice
```

## 📈 Evaluation

Run evaluation with trained checkpoint:

```bash

cd /content/P3D-FusionNet
python eval.py \
    --transformer_config config/3d-retr-b.yaml \
    --annot_path data/ShapeNet.json \
    --model_path $SHAPENET_VOX \
    --image_path $SHAPENET_IMAGES \
    --resume_from_checkpoint "if checkpoint available add the google drive location of the checkpoint into here" \
    --accelerator cpu \
    --save_path /content/predictions \
    --batch_size 16 \
    --num_workers 4 \
    --view_num 1 \
    --split test \
    --save_iou_results
```

🔹 Note: The trained model checkpoints are not uploaded here due to GitHub file size limits. They are included in the Hugging Face Space deployment.

---

## 🎮 Demo
Run the backend locally

Start the backend API by running:

```bash

python app.py

```
This will launch the server at http://localhost:5000.
Test the API locally

Once the backend is running, you can send a request with a 2D image to generate a 3D model.
For example, using curl:

```bash
curl -X POST http://localhost:5000/reconstruct \
  -F "image=@example.png" \
  -o output.obj

```

image → your input 2D image

output.obj → the generated 3D model saved locally

---

## 🙏 Acknowledgements

This project is based on:

[3D-RETR: End-to-End Single and Multi-View 3D Reconstruction with Transformers (BMVC 2021)](https://github.com/fomalhautb/3D-RETR)

---

## 📚 Citation

If you use this work, please cite:

```bash

@inproceedings{3d-retr,
  author    = {Zai Shi, Zhao Meng, Yiran Xing, Yunpu Ma, Roger Wattenhofer},
  title     = {3D-RETR: End-to-End Single and Multi-View 3D Reconstruction with Transformers},
  booktitle = {BMVC},
  year      = {2021}
}

```
## 🧑‍💻 Author

Created by Sunethma Welathanthri


