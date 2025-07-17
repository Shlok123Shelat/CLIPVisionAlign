# CLIPVisionAlign: Image-Caption Alignment Using CLIP

A computer vision project utilizing the CLIP model for precise image-caption alignment on the Flickr30k dataset. This project explores different attention mechanisms—self, cross, and sparse attention—to enhance the alignment of images and their corresponding captions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## Project Overview

This project implements a **CLIP-based model** (Contrastive Language-Image Pretraining) to align images with their captions using the **Flickr30k dataset**. The model leverages both image and text encoders to create embeddings that are projected into a shared space, enabling the alignment of visual and textual data.

The goal is to compare the effectiveness of different attention mechanisms in terms of alignment accuracy and efficiency, providing insights into their suitability for image-caption tasks.

### Key Technologies:
- **Image Encoder**: ResNet50 for extracting visual features from images.
- **Text Encoders**:
  - DistilBert for efficient text representation.
  - Longformer for handling longer text sequences with sparse attention.
- **Attention Variants**:
  - **Self-Attention**: Standard attention mechanism within the text encoder.
  - **Cross-Attention**: Integrates image features directly into the text encoding process.
  - **Sparse Attention**: Uses Longformer’s sparse attention for efficient processing of long captions.

---

## Features

- **Image Encoding**: Utilizes ResNet50 to generate 2048-dimensional image embeddings.
- **Text Encoding**: Supports multiple encoders:
  - DistilBert for standard text encoding.
  - Longformer for sparse attention-based encoding.
- **Custom Attention Mechanisms**:
  - **Self-Attention**: Captures internal text dependencies.
  - **Cross-Attention**: Enables direct interaction between image and text features.
  - **Sparse Attention**: Optimizes memory usage for longer captions.
- **Projection Heads**: Maps image and text embeddings to a shared 256-dimensional space.
- **Training and Validation**: Includes scripts for training and evaluating metrics like loss, accuracy, and recall@5.

---

## Installation and Setup

To set up and run this project locally, follow these steps:

### Prerequisites
- Python 3.8 or higher
- Git
- A CUDA-enabled GPU (recommended for training)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Shlok123Shelat/CLIPVisionAlign.git
   cd CLIPVisionAlign
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   - Ensure you have `pip` installed.
   - Install the required Python packages:
     ```bash
     pip install -r requirements.txt
     ```

4. **Dataset**:
   - Download the **Flickr30k dataset** from [Kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) and place it in `/data/flickr30k_images/`.
   - Update the dataset path in the configuration if using a different location.

5. **Pre-trained Models**:
   - Pre-trained models from `timm` and `transformers` are downloaded automatically when running the script.

---

## Usage

To train and evaluate the model:

### Training the Model
1. **Run the Training Script**:
   ```bash
   python vision_project.py
   ```
   - Trains the model for 2 epochs (configurable in `CFG`) and saves the best model based on validation loss.

2. **Monitor Training**:
   - Uses `tqdm` for progress bars.
   - Outputs training/validation metrics (loss, accuracy, recall@5) per epoch.

### Configuration
- Edit the `CFG` class in `vision_project.py`:
  - `batch_size`: Default 8; adjust based on hardware.
  - `epochs`: Default 2; increase for better results.
  - `device`: Auto-detects GPU if available.

---

## Project Structure

The project structure has been updated based on the repository's current state:

```
CLIPVisionAlign/
│
├── vision_project.py       # Main script for training and evaluating the model
├── requirements.txt        # List of Python dependencies
├── .gitignore              # Git ignore file for unnecessary files
├── README.md               # Project documentation (this file)
├── LICENSE                 # MIT License file
├── data/                   # Directory for dataset (not included; must be downloaded)
│   └── flickr30k_images/   # Flickr30k dataset images and captions
├── Reports/                # Directory for project reports
│   ├── Report_1.pdf        # Initial project report
│   └── Report_2.pdf        # Final project report (renamed from Final_Report.pdf)
```

### Key Files:
- **`vision_project.py`**: Core script with data loading, model, and training logic.
- **`requirements.txt`**: Lists dependencies (e.g., `torch`, `timm`).
- **`Reports/Report_1.pdf`**: Initial project report.
- **`Reports/Report_2.pdf`**: Final project report, renamed from `Final_Report.pdf`.

---

## Results

The project compares three attention variants:
- **Self-Attention**: Balanced performance in accuracy and speed.
- **Cross-Attention**: Highest alignment accuracy but memory-intensive.
- **Sparse Attention**: Most efficient for longer captions.

*Specific metrics (e.g., accuracy %) can be found in the project reports located in the `Reports/` directory.*

---

## Contributing

Contributions are welcome! To contribute:
1. **Fork the Repository**: Click “Fork” on GitHub.
2. **Create a Branch**: Use a descriptive name (e.g., `feature/new-attention`).
3. **Make Changes**: Implement your improvements.
4. **Submit a Pull Request**: Describe your changes in the PR.

Report issues or suggestions via [GitHub Issues](https://github.com/Shlok123Shelat/CLIPVisionAlign/issues).

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **CLIP Model**: OpenAI’s contribution to image-text alignment.
- **Flickr30k Dataset**: University of Illinois dataset for training.
- **Libraries**: Gratitude to `torch`, `timm`, `transformers`, and `albumentations` developers.

---

## Contact

For questions or support:
- Email: [shlokshelat31@gmail.com](mailto:shlokshelat31@gmail.com)
- GitHub: [Shlok123Shelat](https://github.com/Shlok123Shelat)