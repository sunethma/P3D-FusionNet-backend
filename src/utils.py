import io
import os
import torch
import yaml
from PIL import Image
from matplotlib import pyplot as plt
import mlflow
import git

def load_config(config_path):
    with open(config_path) as file:
        return yaml.full_load(file)

def compute_iou(pred, gt):
    pred = pred.clone()
    pred[pred <= 0.5] = 0
    pred[pred >= 0.5] = 1
    intersection = torch.sum(pred.mul(gt)).float()
    union = torch.sum(torch.ge(pred.add(gt), 1)).float()
    return intersection / union

def get_git_revision():
    """Get the current git revision hash."""
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except:
        return None

def get_mlflow_tags():
    """Get MLflow tags including user, git commit, and job ID information."""
    tags = {}
    
    # Get username
    try:
        import getpass
        tags['mlflow.user'] = getpass.getuser()
    except:
        tags['mlflow.user'] = os.environ.get('USER', 'unknown')
    
    # Get git commit
    git_commit = get_git_revision()
    if git_commit:
        tags['mlflow.source.git.commit'] = git_commit
    
    # Get job ID from environment variables
    if 'SLURM_JOB_ID' in os.environ:
        tags['JOB_ID'] = os.environ['SLURM_JOB_ID']
    elif 'LSB_JOBID' in os.environ:
        tags['JOB_ID'] = os.environ['LSB_JOBID']
    
    return tags

def plot_voxels(voxels, rot01=0, rot02=0, rot12=0):
    voxels = voxels[0]
    voxels[voxels >= 0.5] = 1
    voxels[voxels < 0.5] = 0
    voxels = voxels.rot90(rot01, (0, 1))
    voxels = voxels.rot90(rot02, (0, 2))
    voxels = voxels.rot90(rot12, (1, 2))
    
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_box_aspect((1, 1, 1))
    ax.voxels(voxels)
    
    buf = io.BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    
    plt.clf()
    plt.close()
    
    return img