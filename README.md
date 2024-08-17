# Medical Image Classifier for Disease Diagnosis

## Project Overview
This project aims to build a deep learning-based image classifier for medical diagnosis, specifically targeting diseases such as malaria, breast cancer, and lung diseases. The model is trained on labeled medical imaging datasets and deployed as a web application for ease of use by medical practitioners.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Source](#data-source)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Contact Information](#contact-information)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/medical-image-classifier.git
    cd medical-image-classifier
    ```
2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Jupyter Notebooks
For interactive development and visualization, use the Jupyter Notebooks in the `notebooks/` directory. Launch Jupyter Notebook by running:
    ```bash
    jupyter notebook
    ```

### Python Scripts
To run preprocessing, training, and evaluation scripts:
    ```bash
    python scripts/preprocess.py
    python scripts/train.py
    python scripts/evaluate.py
    ```

### Web Application
To start the Flask web application:
    ```bash
    python app/app.py
    ```
Access the application at [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Project Structure
```bash
medical-image-classifier/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   ├── model_evaluation.ipynb
│   └── deployment.ipynb
│
├── scripts/
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── app/
│   ├── app.py
│   └── requirements.txt
│
├── Dockerfile
├── README.md
├── CONTRIBUTING.md
└── LICENSE
```
## Data Source
The datasets used for training and evaluation can be found at the following sources:
- [Malaria Cell Images Dataset](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets)
- [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [Lung Disease Dataset](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data)

Ensure to place the datasets in the `data/raw/` directory before running preprocessing.

## Model Training and Evaluation
Detailed steps for training and evaluating the model are documented in the Jupyter Notebooks:
- `notebooks/data_preprocessing.ipynb`
- `notebooks/model_training.ipynb`
- `notebooks/model_evaluation.ipynb`

## Deployment
The web application is built using Flask. To deploy, follow the steps in the `notebooks/deployment.ipynb` or use the provided Dockerfile for containerized deployment:
```bash
docker build -t medical-image-classifier
docker run -p 5000:5000 medical-image-classifier
```
## Contributing
Contributions are welcome! Please follow these steps to contribute to this project:

1. Fork the repository.
2. Create a new branch for your feature or bugfix:
    ```bash
    git checkout -b feature-name
    ```
3. Make your changes and commit them:
    ```bash
    git commit -m "Description of the feature or fix"
    ```
4. Push to the branch:
    ```bash
    git push origin feature-name
    ```
5. Create a pull request on GitHub.

Please read `CONTRIBUTING.md` for more detailed guidelines on how to contribute to this project.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact Information
For any questions or suggestions, please contact:
- Twilight Techie - [da.kings.air@gmail.com](mailto:da.kings.air@gmail.com).