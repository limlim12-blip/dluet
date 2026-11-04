## Project Structure

```text
├── base/                # Base classes for trainers and data loaders
├── data/                # Dataset directory (ignored in git)
├── logger/              # TensorBoard and logging utilities
├── notebooks/           # Exploratory Jupyter notebooks
├── saved/               # Model checkpoints and evaluation artifacts
├── utils/               # Metric trackers and helper functions
├── config.json          # Main model and training configuration
├── dataset.py           # Dataset parsing and preprocessing logic
├── data_loader.py       # PyTorch DataLoader wrappers
├── main.py              # Entry point for training and evaluation
├── Makefile             # Automation shortcuts for development tasks
├── model.py             # Neural network architecture (1D-TCN / ArcFace)
├── parse_config.py      # Configuration parser utility
├── pyproject.toml       # Project metadata and dependencies
├── sweep.yaml           # Weights & Biases sweep configuration
├── test.py              # Test evaluation script
├── train_atk.py         # Attack training routines
├── train_def.py         # Defense / robust training Trainer class
└── uv.lock              # Dependency lock file (uv)
```

1. Clone the repo 
2. Ensure dependencies are synced:
```Bash
    uv sync
```
3. Download [cesnet-quic22 dataset](https://www.kaggle.com/datasets/zilinpeng/cesnet-quic22). (biggie 80gb+)


### Configuration

Training parameters, model hyperparameters, optimization settings are controlled via config.json.

Key sections include:

    arch: Model architecture configurations.
    trainer: Epoch bounds validation frequencies, and learning rates.
    optimizer: Optimizer settings and learning rate schedulers.

### Usage & Makefile Commands
A Makefile is provided to streamline common development and training operations. Run make help to view all available commands:
### Command	#scription
``` bash
make help				# Show all available Makefile shortcuts and descriptions
make train				# Start model training using main.py
make resume				# Resume training from a saved checkpoint path
make clean				# Remove cache files, build artifacts, and coverage reports
make clean-logs				# Clear local log files
```
(Note: You can adjust the checkpoint path directly inside the Makefile under the resume target).
