import argparse
import os
import json
from pprint import PrettyPrinter

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import ShapeNetDataset, ShuffleDataset, transforms, normalize
from src.data.binvox_rw import Voxels
from src.image2voxel import Image2Voxel
from src.utils import load_config


def save_binvox(voxel, dest, translate, scale):
    binvox = Voxels(voxel, voxel.shape, translate, scale, 'xyz')
    binvox.write(open(dest, 'wb'))


def to_numpy(image):
    image.convert("RGB")
    return [np.asarray(image, dtype=np.float32) / 255]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train transformer conditioned on image inputs')
    parser.add_argument('--annot_path', type=str, required=True,
                        help='Path to "ShapeNet.json"')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the voxel models')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input images')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for dataloader')
    parser.add_argument('--seed', type=int, default=0,
                        help='Manual seed for python, numpy and pytorch')
    parser.add_argument('--split', type=str, default='val',
                        help='"train", "test", or "val"')
    parser.add_argument("--transformer_config", type=str, default=None,
                        help='Path to the image2voxel config file')
    parser.add_argument("--background", type=int, nargs=3, default=(0, 0, 0),
                        help='The (R, G, B) color for the image background')
    parser.add_argument("--beam", type=int, default=1,
                        help='Number of beams for generation')
    parser.add_argument("--view_num", type=int, default=1,
                        help='Number of views for the image input')
    parser.add_argument("--threshold", type=float, default=0.5,
                        help='Threshold for deciding voxel occupancy')
    parser.add_argument("--predict", action='store_true',
                        help='Predict and save results')
    parser.add_argument("--save_path", type=str, default=None,
                        help='Path to save the prediction')
    
    # Add new trainer arguments manually (replacing Trainer.add_argparse_args)
    parser.add_argument("--accelerator", type=str, default=None,
                        help='Accelerator to use (cpu, gpu, tpu, etc.)')
    parser.add_argument("--devices", type=str, default=None,
                        help='Device IDs to use (e.g. 0 or 0,1)')
    parser.add_argument("--num_nodes", type=int, default=1,
                        help='Number of nodes for distributed training')
    parser.add_argument("--precision", type=str, default="32",
                        help='Numerical precision (16, 32, 64, etc.)')
    parser.add_argument("--strategy", type=str, default=None,
                        help='Training strategy for distributed training')
    parser.add_argument("--max_epochs", type=int, default=None,
                        help='Maximum number of epochs')
    parser.add_argument("--min_epochs", type=int, default=None,
                        help='Minimum number of epochs')
    parser.add_argument("--max_steps", type=int, default=-1,
                        help='Maximum number of steps')
    parser.add_argument("--min_steps", type=int, default=None,
                        help='Minimum number of steps')
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument("--save_iou_results", action='store_true',
                        help='Save per-category IoU results to a JSON file')
    parser.add_argument("--iou_results_path", type=str, default="category_iou_results.json",
                        help='Path to save IoU results')

    args = parser.parse_args()

    if args.resume_from_checkpoint is None:
        raise ValueError('No checkpoint specified')

    pp = PrettyPrinter(indent=4)
    pp.pprint(vars(args))

    # =================================================================================

    pl.seed_everything(args.seed)

    image_trans = transforms.Compose([
        to_numpy,
        transforms.CenterCrop((224, 224), (128, 128)),
        transforms.RandomBackground(((240, 240), (240, 240), (240, 240))),
        transforms.ToTensor(),
        lambda x: x[0],
        normalize
    ])

    dataset_params = {
        'annot_path': args.annot_path,
        'model_path': args.model_path,
        'image_path': args.image_path
    }
    dataset = ShapeNetDataset(
        **dataset_params,
        image_transforms=image_trans,
        split=args.split,
        mode='first',
        background=args.background,
        view_num=args.view_num
    )

    dataset = ShuffleDataset(dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )

    # =================================================================================

    transformer_config = load_config(args.transformer_config)
    pp.pprint(transformer_config)
    model = Image2Voxel.load_from_checkpoint(
        threshold=args.threshold,
        checkpoint_path=args.resume_from_checkpoint,
        **transformer_config
    )

    # Extract trainer arguments from args
    trainer_kwargs = {k: v for k, v in vars(args).items() if k in [
        'accelerator', 'devices', 'num_nodes', 'precision', 'strategy',
        'max_epochs', 'min_epochs', 'max_steps', 'min_steps'
    ] and v is not None}
    
    # Create trainer with new API
    trainer = pl.Trainer(logger=False, **trainer_kwargs)

    if args.predict:
        if args.save_path is None:
            raise ValueError('save_path is not specified')

        prediction = trainer.predict(model, loader)
        for pred_dict in tqdm(prediction):
            for i in range(len(pred_dict['generation'])):
                tax_path = os.path.join(args.save_path, pred_dict['taxonomy_id'][i], pred_dict['model_id'][i])

                if not os.path.isdir(tax_path):
                    os.makedirs(tax_path)

                voxel = pred_dict['generation'][i][0].cpu().numpy()
                save_binvox(
                    voxel.astype(bool),
                    os.path.join(tax_path, 'prediction.binvox'),
                    pred_dict['translate'][i].cpu().numpy(),
                    pred_dict['scale'][i].cpu().numpy(),
                )
    else:
        # Test the model
        trainer.test(model, loader)
        
        # # Get and print IoU values for all categories
        # iou_dict = model.get_class_iou(model.test_step_outputs)
        
        # print("\n=== Per-Category IoU Values ===")
        # for category, iou in iou_dict.items():
        #     print(f"{category}: {iou:.4f}")
        
        # # Save IoU results to a file if requested
        # if args.save_iou_results:
        #     with open(args.iou_results_path, 'w') as f:
        #         json.dump(iou_dict, f, indent=4)
        #     print(f"\nCategory IoU results saved to {args.iou_results_path}")