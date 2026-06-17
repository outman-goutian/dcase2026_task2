---
license: mit
tags:
- Audio
- SSL
- EAT
library_name: transformers
base_model:
- worstchan/EAT-base_epoch30_pretrain
---

# EAT-base (Epoch 30, Fine-tuned Checkpoint)

This is the **fine-tuned version** of the [EAT-base (Epoch 30, Pre-trained Checkpoint)](https://huggingface.co/worstchan/EAT-base_epoch30_pretrain), further trained on the AS-2M dataset. Compared to the pre-trained model, this version provides **enhanced audio representations** and typically yields better performance in downstream audio understanding tasks such as classification and captioning.

For more details on the EAT framework, please refer to the [GitHub repository](https://github.com/cwx-worst-one/EAT) and our paper [EAT: Self-Supervised Pre-Training with Efficient Audio Transformer](https://arxiv.org/abs/2401.03497).

## 🔧 Usage

You can load and use the model for feature extraction directly via Hugging Face Transformers:

```python
import torchaudio
import torch
import soundfile as sf
import numpy as np
from transformers import AutoModel

model_id = "worstchan/EAT-base_epoch30_finetune_AS2M"
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval().cuda()

source_file = "/path/to/input.wav"
target_file = "/path/to/output.npy"
target_length = 1024    # Recommended: 1024 for 10s audio
norm_mean = -4.268
norm_std = 4.569

# Load and resample audio
wav, sr = sf.read(source_file)
waveform = torch.tensor(wav).float().cuda()
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)

# Normalize and convert to mel-spectrogram
waveform = waveform - waveform.mean()
mel = torchaudio.compliance.kaldi.fbank(
    waveform.unsqueeze(0),
    htk_compat=True,
    sample_frequency=16000,
    use_energy=False,
    window_type='hanning',
    num_mel_bins=128,
    dither=0.0,
    frame_shift=10
).unsqueeze(0)

# Pad or truncate
n_frames = mel.shape[1]
if n_frames < target_length:
    mel = torch.nn.ZeroPad2d((0, 0, 0, target_length - n_frames))(mel)
else:
    mel = mel[:, :target_length, :]

# Normalize
mel = (mel - norm_mean) / (norm_std * 2)
mel = mel.unsqueeze(0).cuda()  # shape: [1, 1, T, F]

# Extract features
with torch.no_grad():
    feat = model.extract_features(mel)

feat = feat.squeeze(0).cpu().numpy()
np.save(target_file, feat)
print(f"Feature shape: {feat.shape}")
print(f"Saved to: {target_file}")
```

## 📌 Notes

The model supports both **frame-level** (\~50Hz) and **utterance-level** (CLS token) representations.
See the [feature extraction guide](https://github.com/cwx-worst-one/EAT/tree/main/feature_extract) for detailed instructions.


## 📚 Citation

If you find this model useful, please consider citing our [paper](https://arxiv.org/abs/2401.03497):

```bibtex
@article{chen2024eat,
  title={EAT: Self-supervised pre-training with efficient audio transformer},
  author={Chen, Wenxi and Liang, Yuzhe and Ma, Ziyang and Zheng, Zhisheng and Chen, Xie},
  journal={arXiv preprint arXiv:2401.03497},
  year={2024}
}