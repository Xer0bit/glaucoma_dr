# Project Setup Guide

## Prerequisites
- Anaconda or Miniconda installed on your system
- Python 3.x

## Setting up the Environment

1. Clone the repository
```bash
git clone <repository-url>
cd Vaneeza
```

2. Create and activate Conda environment
```bash
# Create new environment
conda create -n vaneeza python=3.9

# Activate environment
# Windows/Linux/MacOS
conda activate vaneeza
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Project Structure
```
Vaneeza/
├── .conda/
├── src/
│   └── main.py
├── requirements.txt
└── README.md
```

## Running the Application
1. Make sure your Conda environment is activated
2. Run the main script:
```bash
python src/main.py
```

## Development

### Managing the Conda environment
If you need to recreate the environment:
1. Remove existing environment:
```bash
conda deactivate
conda env remove -n vaneeza
```
2. Create fresh environment:
```bash
conda create -n vaneeza python=3.8
conda activate vaneeza
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Managing Dependencies
- Add new dependencies to `requirements.txt`
- After adding new packages, update requirements:
```bash
pip freeze > requirements.txt
```

## License
[Add your license information here]

## Contact
[Add your contact information here]
