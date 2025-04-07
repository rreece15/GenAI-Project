# Gen Models project

## Install 

The steps to prepare the environment are outlined below.

### x86-64 Based Systems

1. Create an environment:
```bash
mamba env create -f environments/environment.yaml
```

2. Activate the environment:
```bash
conda activate crecs 
```

### Sign in to Hugging Face

1. Install the Hugging Face CLI (if not installed)
```bash
pip install huggingface_hub
```

2. Log in to Hugging Face
```bash
huggingface-cli login
```

3. Add Hugging Face token


conda_env is just the mac_env converted to conda

