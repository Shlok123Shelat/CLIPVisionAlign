# CLIPVisionAlign: Image-Caption Alignment Using CLIP

A computer vision project utilizing the CLIP model for precise image-caption alignment on the Flickr30k dataset. This project explores different attention mechanisms—self, cross, and sparse attention—to enhance the alignment of images and their corresponding captions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## Project Overview

This project implements a **CLIP-based model** (Contrastive Language-Image Pretraining) to align images with their captions using the **Flickr30k dataset**. The model leverages both image and text encoders to create embeddings that are projected into a shared space, enabling the alignment of visual and textual data.

### Key Technologies:
- **Image Encoder**: ResNet50 for extracting visual features from images.
- **Text Encoders**:
  - DistilBert for efficient text representation.
  - Longformer for handling longer text sequences with sparse attention.
- **Attention Variants**:
  - **Self-Attention**: Standard attention mechanism within the text encoder.
  - **Cross-Attention**: Integrates image features directly into the text encoding process.
  - **Sparse Attention**: Uses Longformer’s sparse attention for efficient processing of long captions.

The project evaluates the performance of these attention variants in terms of alignment accuracy, inference time, and memory usage, providing insights into their effectiveness for image-caption tasks.

---

## Features

- **Image Encoding**: Utilizes ResNet50 to generate 2048-dimensional image embeddings.
- **Text Encoding**: Supports multiple encoders:
  - DistilBert for standard text encoding.
  - Longformer for sparse attention-based encoding.
- **Custom Attention Mechanisms**:
  - **Self-Attention**: For internal text dependencies.
  - **Cross-Attention**: For direct interaction between image and text features.
  - **Sparse Attention**: For efficient handling of longer captions.
- **Projection Heads**: Maps image and text embeddings to a shared 256-dimensional space.
- **Training and Validation**: Includes scripts for training the model and evaluating performance metrics such as loss, accuracy, and recall@5.

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
   git clone https://github.com/yourusername/CLIPVisionAlign.git
   cd CLIPVisionAlign
   ```

2. **Install Dependencies**:
   - Ensure you have `pip` installed.
   - Install the required Python packages:
     ```bash
     pip install -r requirements.txt
     ```

3. **Dataset**:
   - The project uses the **Flickr30k dataset**. Download it from [Kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) and place it in the project directory under `/data/flickr30k_images/`.
   - Alternatively, update the dataset path in the configuration if using a different location.

4. **Pre-trained Models**:
   - The project uses pre-trained models from `timm` and `transformers`. These will be automatically downloaded when the script is run.

---

## Usage

To train and evaluate the model, follow these instructions:

### Training the Model
1. **Run the Training Script**:
   ```bash
   python vision_project.py
   ```
   - This script will train the model for 2 epochs (configurable in `CFG`) and save the best model based on validation loss.

2. **Monitor Training**:
   - The script uses `tqdm` to display progress bars for training and validation loops.
   - Training and validation metrics (loss, accuracy, recall@5) are printed at the end of each epoch.

### Configuration
- The `CFG` class in `vision_project.py` contains hyperparameters and settings. Adjust these as needed:
  - `batch_size`: Set to 8 for stability; increase if your hardware allows.
  - `epochs`: Set to 2 for quick evaluation; increase for better performance.
  - `device`: Automatically detects GPU if available.

---

## Project Structure

The repository is organized as follows:

```
CLIPVisionAlign/
│
├── vision_project.py       # Main script for training and evaluating the model
├── requirements.txt        # List of Python dependencies
├── README.md               # Project documentation (this file)
├── data/                   # Directory for dataset (not included; must be downloaded)
│   └── flickr30k_images/   # Flickr30k dataset images and captions
├── report.pdf              # Project report
```

### Key Files:
- **`vision_project.py`**: Contains the complete code for the CLIP model, including data loading, model definition, training, and evaluation.
- **`requirements.txt`**: Lists all necessary Python packages (e.g., `torch`, `timm`, `transformers`).
- **`report.pdf`**: A detailed report documenting the project’s methodology and results.

---

## Contributing

Contributions are welcome! If you’d like to improve this project, please follow these steps:

1. **Fork the Repository**: Click the “Fork” button on GitHub to create your own copy.
2. **Create a Branch**: Use a descriptive name for your branch (e.g., `feature/new-attention-variant`).
3. **Make Changes**: Implement your improvements or bug fixes.
4. **Submit a Pull Request**: Describe your changes and submit the PR for review.

Please ensure your code follows the project’s style and includes appropriate tests.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

## Acknowledgments

- **CLIP Model**: Developed by OpenAI, used for image-text alignment.
- **Flickr30k Dataset**: Provided by the University of Illinois, used for training and evaluation.
- **Libraries**: Thanks to the developers of `torch`, `timm`, `transformers`, and `albumentations` for their open-source contributions.

---

## Contact

For questions, suggestions, or support, please contact:
- Email: [shlokshelat31@gmail.com](mailto:your.email@example.com)
- GitHub: [Shlok123Shelat](https://github.com/yourusername)

---