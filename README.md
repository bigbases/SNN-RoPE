# Spiking Fused-PE (SF-PE)

**Rotary Positional Embedding (RoPE) for Spiking Neural Networks**

This repository implements Rotary Positional Embedding (RoPE) for Spiking Neural Networks (SNNs), combined with Central Pattern Generator (CPG) based positional encoding.

## Key Contributions

### RoPE for SNN
- Multiple RoPE variants for temporal and spatial dimensions:
  - `RotaryEmbedding1DTemporal`: Temporal dimension rotary embedding
  - `RotaryEmbedding1DSpatial`: Spatial dimension rotary embedding  
  - `RotaryEmbedding2D`: Combined spatiotemporal rotary embedding
- Spiking-optimized rotary positional embedding application

## Repository Structure

```
SNN-RoPE/
├── README.md                 # This file
├── spikformer_cpg_rope.py   # Main Spikformer CPG-RoPE model
├── CPG.py                   # Central Pattern Generator implementation
├── CPG_RoPE_hybrid.py       # RoPE variants and hybrid implementation
└── .git/                    # Git repository
```

## Training with SeqSNN

This repository's models can be trained using the **SeqSNN framework**:

### Setup SeqSNN Environment

```bash
# Install SeqSNN
conda create -n SeqSNN python=3.9
conda activate SeqSNN
git clone https://github.com/microsoft/SeqSNN/
cd SeqSNN
pip install -e .
```

### File Integration Guide

To use SNN-RoPE models with SeqSNN, you need to place files in specific directories:

#### 1. Model Files (Python modules)
Copy the model implementation to SeqSNN's network directory:

```bash
# Copy model implementation files
cp /path/to/SNN-RoPE/spikformer_cpg_rope.py /path/to/SeqSNN/SeqSNN/network/snn/
cp /path/to/SNN-RoPE/CPG_RoPE_hybrid.py /path/to/SeqSNN/SeqSNN/module/
cp /path/to/SNN-RoPE/CPG.py /path/to/SeqSNN/SeqSNN/module/
```

#### 2. Update Network Init File
**Important**: Add the model import to SeqSNN's network `__init__.py` file:

Edit `/path/to/SeqSNN/SeqSNN/network/__init__.py` and add:

```python
from .base import NETWORKS

# ... existing imports ...
from .snn.spikformer_cpg_rope import SpikformerCPGRoPE  # ← Add this line
```

**Complete example of `__init__.py`:**
```python
from .base import NETWORKS

from .ann.tsrnn import TSRNN
from .ann.itransformer import ITransformer
from .ann.tcn2d import TemporalConvNet2D, TemporalBlock2D

from .snn.snn import TSSNN, TSSNN2D, ITSSNN2D
from .snn.spike_tcn import SpikeTemporalConvNet2D, SpikeTemporalBlock2D
from .snn.ispikformer import iSpikformer
from .snn.spikformer import Spikformer
from .snn.spikformer_cpg_rope import SpikformerCPGRoPE  # ← RoPE model import
```

**Directory structure in SeqSNN:**
```
SeqSNN/
├── SeqSNN/
│   ├── network/
│   │   ├── __init__.py                   # ← Add import here!
│   │   └── snn/
│   │       └── spikformer_cpg_rope.py    # ← Model implementation
│   └── module/
│       ├── CPG_RoPE_hybrid.py            # ← RoPE variants
│       └── CPG.py                        # ← CPG implementation
└── exp/
    └── forecast/
        └── spikformer/                    # ← Configuration files
            ├── CPGRoPE_electricity.yml
            └── test.yml
```

#### 3. Configuration Files (YAML)
Copy experiment configuration files to the appropriate forecast directory:

```bash
# Copy configuration files to SeqSNN repository
cp /path/to/SNN-RoPE/exp/*.yml /path/to/SeqSNN/exp/forecast/spikformer/

# Or create a dedicated directory for RoPE experiments
mkdir -p /path/to/SeqSNN/exp/forecast/spikformer_rope/
cp /path/to/SNN-RoPE/exp/*.yml /path/to/SeqSNN/exp/forecast/spikformer_rope/
```

#### 4. Running Training

After placing the files correctly, run training:

```bash
# Navigate to SeqSNN directory
cd /path/to/SeqSNN

# Example: Train Spikformer with CPG-RoPE on electricity dataset
python -m SeqSNN.entry.tsforecast exp/forecast/spikformer/CPGRoPE_electricity.yml

# Or if you created a dedicated rope directory:
python -m SeqSNN.entry.tsforecast exp/forecast/spikformer_rope/spikformer_rope_electricity.yml
```

### Configuration File Structure

The YAML configuration files include:
- **Dataset settings**: Window size, horizon, normalization
- **Model parameters**: Dimensions, depths, attention heads
- **RoPE settings**: `use_rope: True`, `rope_theta`, `rope_mode`
- **Training settings**: Learning rate, batch size, epochs

#### Important: Network Type Registration

Make sure the network type matches the registered name in the model file:

```python
@NETWORKS.register_module("SpikformerCPGRoPE")  # ← This name must match YAML
class SpikformerCPGRoPE(nn.Module):
    # ... model implementation
```

#### Example Configuration Files

**Complete configuration example (`CPGRoPE_electricity.yml`):**
```yaml
_base_:
- ../dataset/electricity.yml
_custom_imports_:
- SeqSNN.network
- SeqSNN.dataset
- SeqSNN.runner

data:
  raw_label: False
  window: 6
  horizon: 96
  normalize: 3

runner:
  type: ts
  task: regression
  optimizer: Adam
  lr: 0.001
  batch_size: 64
  max_epoches: 1000

network:
  type: SpikformerCPGRoPE          # ← Must match registered name
  dim: 256
  d_ff: 1024
  depths: 2
  num_steps: 4
  heads: 8
  # RoPE specific parameters
  use_rope: True
  rope_theta: 10000.0
  rope_mode: 2d                     # Options: "2d", "1d_temporal", "1d_spatial"

runtime:
  seed: 41
  output_dir: ./outputs/test
```

#### RoPE Mode Options

- **`2d`**: Combined spatiotemporal rotary embedding (default)
- **`1d_temporal`**: Temporal dimension only
- **`1d_spatial`**: Spatial dimension only

### Troubleshooting

#### Common Issues and Solutions

1. **Module Not Found Error**
   ```bash
   ModuleNotFoundError: No module named 'SeqSNN.network.snn.spikformer_cpg_rope'
   ```
   **Solution**: Ensure model files are copied to correct locations AND imported in `__init__.py`:
   - Copy files: `spikformer_cpg_rope.py` → `SeqSNN/SeqSNN/network/snn/`
   - Copy modules: `CPG_RoPE_hybrid.py`, `CPG.py` → `SeqSNN/SeqSNN/module/`
   - **Add import** in `SeqSNN/SeqSNN/network/__init__.py`:
     ```python
     from .snn.spikformer_cpg_rope import SpikformerCPGRoPE
     ```

2. **Import Error (Model not accessible)**
   ```bash
   AttributeError: module 'SeqSNN.network' has no attribute 'SpikformerCPGRoPE'
   ```
   **Solution**: Model is copied but not imported. Add to `SeqSNN/SeqSNN/network/__init__.py`:
   ```python
   from .snn.spikformer_cpg_rope import SpikformerCPGRoPE
   ```

2. **Network Type Not Found**
   ```bash
   KeyError: 'SpikformerCPGRoPE'
   ```
   **Solution**: Check that the model file is imported correctly and the `@NETWORKS.register_module()` decorator matches the YAML `type` field.


## Related Papers

* **Efficient and Effective Time-Series Forecasting with Spiking Neural Networks**, ICML 2024
  - [arXiv:2402.01533](https://arxiv.org/pdf/2402.01533)
  
* **Advancing Spiking Neural Networks for Sequential Modeling with Central Pattern Generators**, NeurIPS 2024  
  - [arXiv:2405.14362](https://arxiv.org/pdf/2405.14362)

## License

This project is built upon Microsoft's SeqSNN framework. Please refer to the [SeqSNN repository](https://github.com/microsoft/SeqSNN/) for detailed contribution guidelines and license information.
