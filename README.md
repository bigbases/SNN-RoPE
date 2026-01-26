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

### Using Configuration Files

This repository provides pre-configured YAML files for training different models. Copy the configuration files from the `exp/` directory to the SeqSNN repository:

```bash
# Copy configuration files to SeqSNN repository
cp /path/to/SNN-RoPE/exp/*.yml /path/to/SeqSNN/exp/forecast/spikformer_rope/

# Example: Train Spikformer with RoPE on electricity dataset
python -m SeqSNN.entry.tsforecast exp/forecast/spikformer_rope/spikformer_rope_electricity.yml
```

### Configuration File Structure

The YAML configuration files include:
- **Dataset settings**: Window size, horizon, normalization
- **Model parameters**: Dimensions, depths, attention heads
- **RoPE settings**: `use_rope: True`, `rope_theta`, `rope_mode`
- **Training settings**: Learning rate, batch size, epochs

Example configuration for electricity forecasting:
```yaml
network:
  type: SpikformerCPGRoPE
  dim: 256
  d_ff: 1024
  depths: 2
  num_steps: 4
  heads: 8
  # RoPE specific parameters
  use_rope: True
  rope_theta: 10000.0
  rope_mode: 2d
```

## Related Papers

* **Efficient and Effective Time-Series Forecasting with Spiking Neural Networks**, ICML 2024
  - [arXiv:2402.01533](https://arxiv.org/pdf/2402.01533)
  
* **Advancing Spiking Neural Networks for Sequential Modeling with Central Pattern Generators**, NeurIPS 2024  
  - [arXiv:2405.14362](https://arxiv.org/pdf/2405.14362)

## License

This project is built upon Microsoft's SeqSNN framework. Please refer to the [SeqSNN repository](https://github.com/microsoft/SeqSNN/) for detailed contribution guidelines and license information.
