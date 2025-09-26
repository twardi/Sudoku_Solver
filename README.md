# Genetic Sudoku Solver

## Requirements

- Python 3.8 or higher
- Required packages: numpy, pygad, matplotlib

## Installation

### Option 1: Using pip (Recommended)

1. **Clone or download the project**
   ```bash
   cd Sudoku_Solver
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using Conda

1. **Create conda environment**
   ```bash
   conda create -n sudoku_solver python=3.10
   conda activate sudoku_solver
   ```

2. **Install dependencies**
   ```bash
   conda install numpy matplotlib
   pip install pygad  # PyGAD is not available on conda-forge, use pip
   ```

3. **Alternative: Use pip within conda environment**
   ```bash
   conda activate sudoku_solver
   pip install -r requirements.txt
   ```

4. **Export environment (optional)**
   ```bash
   # Create environment file for sharing
   conda env export > environment.yml
   
   # Others can recreate with:
   # conda env create -f environment.yml
   ```

## Usage

### Basic Usage

Run the solver with default configuration:

```bash
python sudoku.py
```