# YOLOv7 Object Detection Project

This project implements object detection based on YOLOv7, containing two models:
- **YOLOv7 Official Pre-trained Model**: A general-purpose object detection model trained on COCO dataset (no training required, use directly)
- **Face Mask Detection Custom Model**: A custom model trained on face mask dataset (face_mask_dataset)

## Project Structure

```
yolov7-main/
├── detect.py              # Inference script
├── train.py               # Training script
├── test.py                # Testing script
├── requirements.txt       # Dependencies list
├── data/                  # Dataset configuration files
│   ├── coco.yaml          # COCO dataset configuration (for testing official model)
│   └── face_mask.yaml     # Face mask dataset configuration file
├── face_mask_dataset/     # Face mask detection dataset
│   ├── Images/            # Image folder
│   ├── Annotations/       # XML annotation files
│   ├── labels/            # YOLO format label files
│   ├── train.txt          # Training set list
│   └── val.txt            # Validation set list
├── models/                # Model definitions
├── utils/                 # Utility functions
├── runs/                  # Run results
│   ├── train/             # Training results
│   │   └── exp/
│   │       └── weights/
│   │           └── best.pt  # Best model trained on face mask dataset
│   ├── detect/            # Inference results
│   └── test/              # Testing results
└── inference/             # Inference test images
    └── images/
```

## Requirements

- Python >= 3.7
- PyTorch >= 1.7.0
- CUDA (optional, for GPU acceleration)

## Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

### Main Dependencies

**Base Libraries:**
- `torch>=1.7.0,!=1.12.0` - PyTorch deep learning framework
- `torchvision>=0.8.1,!=0.13.0` - Computer vision toolkit
- `opencv-python>=4.1.1` - Image processing library
- `numpy>=1.18.5,<1.24.0` - Numerical computing library
- `Pillow>=7.1.2` - Image processing library
- `PyYAML>=5.3.1` - YAML file parser
- `matplotlib>=3.2.2` - Plotting library

**Training Related:**
- `tensorboard>=2.4.1` - Training visualization tool
- `pandas>=1.1.4` - Data processing
- `seaborn>=0.11.0` - Statistical plotting

**Utility Libraries:**
- `tqdm>=4.41.0` - Progress bar
- `scipy>=1.4.1` - Scientific computing
- `thop` - FLOPs computation
- `psutil` - System resource monitoring

## Model Description

### 1. YOLOv7 Official Pre-trained Model (COCO Dataset)

- **Model File**: `yolov7.pt` (needs to be downloaded from official source, will be downloaded automatically on first run)
- **Dataset**: COCO 2017
- **Number of Classes**: 80 classes
- **Purpose**: General-purpose object detection, can detect 80 common objects
- **Note**: This is the official pre-trained model provided by YOLOv7, **no training required**, can be used directly for inference and testing

### 2. Face Mask Detection Model (Custom Training)

- **Model File**: `runs/train/exp/weights/best.pt`
- **Dataset**: `face_mask_dataset` (face mask detection dataset)
- **Configuration File**: `data/face_mask.yaml`
- **Number of Classes**: 3 classes (face, face_mask, mask)
- **Purpose**: Specifically designed for detecting faces, masked faces, and masks
- **Note**: This is a custom model trained on the `face_mask_dataset` using the `data/face_mask.yaml` configuration file

## Usage

### 1. Training

**Note**: This project only trains the face mask detection model. The COCO model uses the YOLOv7 official pre-trained model and requires no training.

#### Training Face Mask Detection Model

Train the face mask detection model using the `face_mask_dataset` dataset and `data/face_mask.yaml` configuration file:

```bash
python train.py --weights yolov7.pt --data data/face_mask.yaml --hyp data/hyp.scratch.p5.yaml --epochs 300 --batch-size 8 --img-size 640
```

**Main Parameters:**
- `--weights`: Pre-trained weights path (default: `yolov7.pt`, uses YOLOv7 official pre-trained model as starting point)
- `--data`: Dataset configuration file path (**required**: `data/face_mask.yaml`)
- `--hyp`: Hyperparameter configuration file path (default: `data/hyp.scratch.p5.yaml`)
- `--epochs`: Number of training epochs (default: 300)
- `--batch-size`: Batch size (default: 8, adjust according to GPU memory)
- `--img-size`: Input image size (default: [640, 640])
- `--device`: Specify device, e.g., `--device 0` for GPU 0, or `--device cpu` for CPU
- `--project`: Save path (default: `runs/train`)
- `--name`: Experiment name (default: `exp`)

**Dataset Information:**
- Dataset directory: `face_mask_dataset/`
- Dataset configuration: `data/face_mask.yaml`
- Ensure the dataset path is correctly configured in `face_mask.yaml`

**Training Results Save Location:**
- Best model: `runs/train/exp/weights/best.pt`
- Latest model: `runs/train/exp/weights/last.pt`
- Training log: `runs/train/exp/results.txt`
- TensorBoard log: `runs/train/exp/`

**View Training Process:**
```bash
tensorboard --logdir runs/train
```
Then open `http://localhost:6006` in your browser

### 2. Testing

#### Testing Face Mask Detection Model

```bash
python test.py --weights runs/train/exp/weights/best.pt --data data/face_mask.yaml --batch-size 32 --img-size 640
```

**Main Parameters:**
- `--weights`: Model weights path
- `--data`: Dataset configuration file path
- `--batch-size`: Batch size (default: 32)
- `--img-size`: Input image size (default: 640)
- `--conf-thres`: Confidence threshold (default: 0.001)
- `--iou-thres`: IOU threshold (default: 0.65)
- `--task`: Task type, options: `train`, `val`, `test` (default: `val`)
- `--save-txt`: Save detection results as txt files
- `--save-json`: Save results in COCO format JSON
- `--verbose`: Display detailed results for each class

**Test Results Save Location:**
- Result images: `runs/test/exp/`
- Confusion matrix: `runs/test/exp/confusion_matrix.png`
- Performance curves: `runs/test/exp/*_curve.png`

### 3. Inference

#### Using Face Mask Detection Model for Inference

```bash
python detect.py --weights runs/train/exp/weights/best.pt --source inference/images --img-size 640 --conf-thres 0.25
```

#### Using COCO Official Pre-trained Model for Inference

```bash
python detect.py --weights yolov7.pt --source inference/images --img-size 640 --conf-thres 0.25
```

**Note**: The COCO model is the YOLOv7 official pre-trained model and will be downloaded automatically on first use.

**Main Parameters:**
- `--weights`: Model weights path (default: `yolov7.pt`)
- `--source`: Input source, can be:
  - Image path: `inference/images/`
  - Video path: `path/to/video.mp4`
  - Webcam: `0` (use default camera)
  - Network stream: `rtsp://...` or `http://...`
- `--img-size`: Input image size (default: 640)
- `--conf-thres`: Confidence threshold (default: 0.25)
- `--iou-thres`: IOU threshold (default: 0.45)
- `--device`: Specify device (default: `0`, use GPU 0)
- `--view-img`: Display detection results
- `--save-txt`: Save detection results as YOLO format txt files
- `--save-conf`: Save confidence in txt files
- `--nosave`: Do not save detection result images
- `--project`: Save path (default: `runs/detect`)
- `--name`: Experiment name (default: `exp`)

**Inference Examples:**

1. **Detect a single image:**
```bash
python detect.py --weights runs/train/exp/weights/best.pt --source inference/images/1.jpg --view-img
```

2. **Detect all images in a folder:**
```bash
python detect.py --weights runs/train/exp/weights/best.pt --source inference/images/ --save-txt
```

3. **Real-time detection using webcam:**
```bash
python detect.py --weights runs/train/exp/weights/best.pt --source 0 --view-img
```

4. **Detect video file:**
```bash
python detect.py --weights runs/train/exp/weights/best.pt --source path/to/video.mp4
```

**Inference Results Save Location:**
- Detection result images: `runs/detect/exp/`
- Label files (if using --save-txt): `runs/detect/exp/labels/`

## Quick Start Example

### Complete Workflow Example (Face Mask Detection)

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Train model:**
```bash
python train.py --weights yolov7.pt --data data/face_mask.yaml --epochs 300 --batch-size 8
```

3. **Test model:**
```bash
python test.py --weights runs/train/exp/weights/best.pt --data data/face_mask.yaml
```

4. **Run inference:**
```bash
python detect.py --weights runs/train/exp/weights/best.pt --source inference/images/ --view-img
```

## Notes

1. **GPU Memory Insufficient**: If you encounter out of memory errors, reduce the `--batch-size` parameter
2. **Model Download**: The YOLOv7 official pre-trained model `yolov7.pt` needs to be downloaded on first use, it will be downloaded automatically
3. **Dataset Path**: Ensure the face mask dataset path is correctly configured, check the path settings in `data/face_mask.yaml` to ensure it points to the `face_mask_dataset` directory
4. **Device Selection**: If you don't have a GPU, use the `--device cpu` parameter, but inference speed will be slower
5. **COCO Model**: The COCO model uses the YOLOv7 official pre-trained model, no training required, use directly

## FAQ

**Q: CUDA out of memory error during training?**  
A: Reduce the `--batch-size` parameter, or reduce the `--img-size` parameter

**Q: How to resume training an interrupted model?**  
A: Use the `--resume` parameter:
```bash
python train.py --resume runs/train/exp/weights/last.pt
```

**Q: How to test without training?**  
A: Use the `--notest` parameter to skip testing during training, or use the `test.py` script directly

**Q: How to adjust detection thresholds?**  
A: Use `--conf-thres` to adjust confidence threshold, use `--iou-thres` to adjust IOU threshold

## License

Please refer to the LICENSE.md file in the project
