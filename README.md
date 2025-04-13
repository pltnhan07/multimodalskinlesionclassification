## My Project: ConvNeXt-CBAM Classifier with Metadata Fusion

This project implements a convolutional neural network for image classification that fuses image data with metadata.
It uses a ConvNeXt backbone augmented with CBAM modules. The model is trained using a custom Focal Loss.

### Project Structure

- **data/**: Contains CSV files for metadata and image subfolders.
- **src/**: Source code for dataset handling, model definition, training, loss functions, and utility functions.
- **config.yaml**: Configuration file for hyperparameters, file paths, and other settings.
- **requirements.txt**: A list of required Python packages.

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd my_project
   
2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # (or `venv\Scripts\activate` on Windows)
   pip install -r requirements.txt
   
3. **Configure the project**

4. **Run Training**
   ```bash
   python src/train.py

