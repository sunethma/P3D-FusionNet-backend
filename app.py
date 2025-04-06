import os
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.responses import Response
import io
import tempfile
import shutil

# Import your project modules
from src.image2voxel import Image2Voxel
from src.data.binvox_rw import Voxels
from src.data import normalize
from src.utils import load_config
# Import the transforms
from src.data.transforms import CenterCrop, Normalize, ToTensor

app = FastAPI()

# Load the model at startup to avoid loading it on each request
MODEL = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIG_PATH = "config/3d-retr-b.yaml"
CHECKPOINT_PATH = "model_checkpoint/checkpoint.ckpt"  # Update this to your model path

def apply_background(img_batch, background_color=(240, 240, 240)):
    """
    Apply a solid background color to images in the batch.
    This is a simplified version of the RandomBackground used in training.
    
    Args:
        img_batch: Numpy array of shape [batch_size, height, width, channels]
        background_color: RGB tuple for background color (default: (240, 240, 240))
    
    Returns:
        Processed image batch with solid background
    """
    if len(img_batch) == 0:
        return img_batch
        
    img_height, img_width, img_channels = img_batch[0].shape
    
    # If the image doesn't have alpha channel, return as is
    if not img_channels == 4:
        return img_batch
    
    # Convert background color to float in range [0, 1]
    r, g, b = np.array(background_color) / 255.0
    
    # Apply background
    processed_images = np.empty(shape=(0, img_height, img_width, img_channels - 1))
    for img_idx, img in enumerate(img_batch):
        # Create alpha mask (1 where background should be, 0 elsewhere)
        alpha = (np.expand_dims(img[:, :, 3], axis=2) == 0).astype(np.float32)
        
        # Remove alpha channel
        img = img[:, :, :3]
        
        # Apply background color where alpha indicates
        bg_color = np.array([[[r, g, b]]])
        img = alpha * bg_color + (1 - alpha) * img
        
        # Add to processed images
        processed_images = np.append(processed_images, [img], axis=0)
    
    return processed_images

def preprocess_image(image_data, target_size=(224, 224), crop_size=(128, 128), background=(240, 240, 240)):
    """Load and preprocess image for the model."""
    # Load image from bytes
    image = Image.open(io.BytesIO(image_data))
    
    # Convert PIL to numpy array
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_array = np.asarray(image, dtype=np.float32) / 255
    
    # Create a batch of one image (required by CenterCrop)
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Apply center crop transform - using the same crop sizes as in training
    center_crop = CenterCrop(target_size, crop_size)
    img_batch = center_crop(img_batch)
    
    # Apply background transform
    img_batch = apply_background(img_batch, background)
    
    # Since we have a batch of 1, get the single image back
    img_array = img_batch[0]
    
    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC to CHW
    
    # Apply normalization from data module
    img_tensor = normalize(img_tensor)
    
    # Ensure tensor is float32, not double (float64)
    img_tensor = img_tensor.float()
    
    return img_tensor

def load_model(checkpoint_path, config_path=None, threshold=0.5):
    """Load the Image2Voxel model from checkpoint."""
    if config_path:
        config = load_config(config_path)
    else:
        config = {}
    
    model = Image2Voxel.load_from_checkpoint(
        threshold=threshold,
        checkpoint_path=checkpoint_path,
        **config
    )
    
    model.eval()
    return model

def generate_voxel(model, image_tensor, threshold=0.5, beam=1):
    """Generate voxel prediction from image tensor."""
    # Add batch dimension if not present
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    # Prepare batch dictionary as expected by the model
    batch = {'image': image_tensor}
    
    # Move to the same device as model
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # Generate voxel with no_grad
    with torch.no_grad():
        voxel = model.generate(batch, temperature=1.0, sample=False, beam=beam)
    
    # Apply threshold
    voxel = voxel.cpu().numpy()
    voxel = (voxel > threshold).astype(np.float32)
    
    return voxel

def save_binvox(voxel, dest_path, translate=[0.0, 0.0, 0.0], scale=32.0):
    """Save numpy voxel array as binvox file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    
    # Convert to boolean and create Voxels object
    voxel_bool = voxel.astype(np.bool_)
    binvox = Voxels(voxel_bool, voxel_bool.shape, translate, scale, 'xyz')
    
    # Write to file
    with open(dest_path, 'wb') as f:
        binvox.write(f)

@app.on_event("startup")
def startup_event():
    global MODEL
    MODEL = load_model(CHECKPOINT_PATH, CONFIG_PATH)
    MODEL = MODEL.to(DEVICE)
    print(f"Model loaded on {DEVICE}")

@app.get("/")
def read_root():
    return {"message": "3D-RETR API is running. Send POST requests to /generate with an image file"}

@app.post("/generate")
async def generate_model(file: UploadFile = File(...), threshold: float = 0.5, beam: int = 1, background: str = "240,240,240"):
    try:
        # Process the background color
        bg_color = tuple(map(int, background.split(',')))
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Read the uploaded file
            image_data = await file.read()
            
            # Preprocess the image
            image_tensor = preprocess_image(image_data, background=bg_color)
            image_tensor = image_tensor.to(DEVICE)
            
            # Generate voxel
            voxel = generate_voxel(MODEL, image_tensor, threshold, beam)
            
            # Extract first item from batch
            voxel = voxel[0, 0]
            
            # Save as binvox
            output_path = os.path.join(temp_dir, "model.binvox")
            save_binvox(voxel, output_path)
            
            # Read the binvox file into memory before the temp dir is cleaned up
            with open(output_path, 'rb') as f:
                binvox_data = f.read()
            
        # Return the binary data directly instead of as a file
        return Response(
            content=binvox_data,
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=model.binvox"}
        )
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)