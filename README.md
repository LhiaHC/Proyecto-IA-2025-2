# Email Spam and Phishing Classifier

This project implements and evaluates a machine learning pipeline for classifying emails as spam/phishing or legitimate. It systematically experiments with different text embedding techniques and classification models to find the most effective combination.

The entire workflow is automated with shell scripts and designed for efficiency, featuring automatic GPU detection, data preprocessing, and results aggregation.

## Key Features

-   **Multiple Embedding Strategies**: Compares the performance of Word2Vec, FastText, and fine-tuned BERT embeddings.
-   **Multiple Classifier Models**: Evaluates Logistic Regression, Support Vector Machines (SVM), and Random Forest.
-   **Automatic GPU Acceleration**: Intelligently detects if a compatible NVIDIA GPU is available and uses the RAPIDS `cuml` library for massive speedups.
-   **Graceful CPU Fallback**: If no GPU is found, the project seamlessly falls back to using `scikit-learn` on the CPU, making the code portable to any machine.
-   **Efficient Data Pipeline**:
    1.  **Preprocess Once**: A single script processes the raw text data.
    2.  **Cache Embeddings**: Generated BERT embeddings are automatically saved to disk to avoid redundant, time-consuming calculations on subsequent runs.
-   **Robust Evaluation**: Uses stratified k-fold cross-validation to ensure reliable performance metrics.
-   **Automated Experimentation**: A shell script (`run_all_experiments.sh`) orchestrates the entire evaluation process, running all 30+ experimental combinations and saving the results.

## Project Structure

```
.
├── data/
│   ├── raw/                 # Original, untouched datasets
│   ├── processed/           # Cleaned data ready for modeling (created by scripts)
│   └── embeddings/          # Cached BERT embeddings (created by scripts)
├── reports/                 # Output CSV files with experiment results
├── src/                     # All Python source code
│   ├── create_sample.py     # Script to generate a smaller sample dataset
│   ├── preprocess_data.py   # Script to clean text and prepare data for modeling
│   ├── experiment_runner.py # The core script that runs a single experiment
│   ├── utils_00.py          # Utility functions
│   ├── preproc_01.py        # Text preprocessing functions
│   └── embeddings_03.py     # Embedding generation classes
├── notebooks/               # Jupyter notebooks for exploration and analysis
├── requirements.txt         # Core dependencies for CPU execution
├── requirements-gpu.txt     # Optional dependencies for GPU acceleration
└── run_all_experiments.sh   # Main script to run all experiments
```

## Installation

Follow these steps to set up the project environment.

### 1. Clone the Repository

```bash
git clone https://github.com/LhiaHC/Proyecto-IA-2025-2.git
cd Proyecto-IA-2025-2
```

### 2. Create a Python Virtual Environment

This isolates the project dependencies.

```bash
python -m venv iaa_venv
source iaa_venv/bin/activate
# On Windows, use: iaa_venv\Scripts\activate
```

### 3. Install Dependencies

Choose one of the following two paths depending on your hardware.

#### Path A: CPU-Only Setup (Works on any computer)

This installs the core libraries. The scripts will run correctly on your CPU.

```bash
pip install -r requirements.txt
```

#### Path B: GPU-Accelerated Setup (For NVIDIA GPUs)

This path enables massive performance gains.

**First, install the core CPU dependencies:**

```bash
pip install -r requirements.txt
```

**Next, install the GPU libraries.** These require having the NVIDIA drivers and a compatible CUDA Toolkit installed on your system.

```bash
pip install -r requirements-gpu.txt
```
> **Note on GPU Installation**: Installing `cuml` and `cupy` with `pip` can sometimes be tricky due to system dependencies. If you encounter issues, the most stable installation method recommended by the library authors is using `conda`. You can find the specific command for your system on the [RAPIDS installation page](https://rapids.ai/start.html).

## How to Run the Full Experiment Suite

The project is designed around a simple, three-step workflow.

### Step 1: Create a Data Sample (Run Once)

This script takes the original large dataset and creates a smaller, 5,000-row sample for faster experimentation.

```bash
python src/create_sample.py
```

### Step 2: Preprocess the Sample Data (Run Once)

This script takes the sample created in the previous step, applies all the heavy text cleaning and preprocessing, and saves a final, clean Parquet file. This is the file that all experiments will use.

```bash
python src/preprocess_data.py
```

### Step 3: Run All Experiments

This is the main step. The shell script will automatically iterate through all combinations of embeddings, dimensions, and classifiers, printing the results and saving a summary CSV for each in the `reports/` directory.

The first time you run experiments with BERT, it will generate and cache the embeddings. All subsequent runs will be significantly faster as they will load from the cache.

```bash
# Make the script executable (you only need to do this once)
chmod +x run_all_experiments.sh

# Execute the script
./run_all_experiments.sh
```

After the script finishes, the `reports/` directory will be populated with the results of every experiment.
