# Self-Supervised Visual Representations

Educational PyTorch implementations of **SimCLR**, **masked autoencoding (MAE)** and **MoCo v2**, with image pretraining followed by frozen-backbone linear-probe evaluation.

The experiments use an ImageNet-100-style dataset. Default pretraining is deliberately short (5 epochs), so these runs demonstrate the pipelines rather than establish competitive benchmark performance.

## Implementations

| Entry point | Method | Backbone | Default pretraining / linear probe |
| --- | --- | --- | --- |
| `run_simclr.py` | Two-view contrastive learning, NT-Xent loss | ResNet-18 | 5 / 10 epochs |
| `run_mae.py` | Masked patch reconstruction | ViT-Base/16 | 5 / 5 epochs |
| `run_moco.py` | Momentum encoder and negative queue | ResNet-18 | 5 / 5 epochs |
| `run_ssl_project.py` | Earlier SimCLR runner | ResNet-18 | Loads an existing backbone by default |

SimCLR uses a 128-dimensional projection head, batch size 32 and temperature 0.07. MoCo uses a queue of 16,384 embeddings, momentum 0.999 and temperature 0.07. MAE masks 75% of patches and uses a 512-dimensional, 8-layer decoder. See each script's configuration block for the complete settings.

## Setup

Use a Python version supported by your selected PyTorch/torchvision builds; Python 3.10 or 3.11 is a practical starting point. Install a mutually compatible PyTorch/torchvision pair for your hardware.

```sh
git clone https://github.com/Lakshaypal/self-supervised-visual-representations.git
cd self-supervised-visual-representations
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

CUDA, Apple MPS and CPU are selected automatically by the scripts. Training ViT-Base is substantially more demanding than running the ResNet examples.

## Dataset layout

Prepare your own authorised dataset outside Git:

```text
ssl_dataset/
  Labels.json
  train.X1/<class-id>/*.JPEG
  train.X2/<class-id>/*.JPEG
  val.X/<class-id>/*.JPEG
```

Training folders may use any names beginning with `train.X`. `Labels.json` must be a JSON object mapping class IDs to names. Keep class-directory names and their class-to-index ordering consistent across every train shard and validation.

Set the dataset root before running:

```sh
export SSL_DATA_PATH=/absolute/path/to/ssl_dataset
python run_simclr.py
python run_mae.py
python run_moco.py
```

Without `SSL_DATA_PATH`, the scripts look in `data/ssl_dataset` beside the source files. Run one script at a time. Hyperparameters and the `SKIP_*_PRETRAINING` switches are configured inside the scripts; the earlier `run_ssl_project.py` requires its SimCLR checkpoint unless you turn pretraining on.

## Outputs

Training writes checkpoint files and loss/accuracy/F1 plots into the working directory. Historical figures are in [`plots/`](plots/), and the project report is [`Lakshay_Pal_SSL_Report.pdf`](Lakshay_Pal_SSL_Report.pdf). Checkpoints and the training dataset are not bundled in this repository.

A decreasing pretraining loss is not classification accuracy. Use the linear-probe validation accuracy and macro-F1, together with the dataset split and training budget, when comparing runs.

## Licence

[MIT](LICENSE).
