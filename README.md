# Safety Operator — Open Source AI "Seatbelt"

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxx)
[![GitHub stars](https://img.shields.io/github/stars/DerekEarnhart/safety-operator?style=social)](https://github.com/DerekEarnhart/safety-operator/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/DerekEarnhart/safety-operator?style=social)](https://github.com/DerekEarnhart/safety-operator/network/members)

---

A **drop‑in safety filter** for AI model outputs.  
Projects outputs into a defined safe region, dampens unsafe components, and applies optional alignment regularization — *without touching model internals*.  
Free, open source, and ready to integrate with any model.

---

## 🚀 Features

- **Model‑agnostic** — works with logits, embeddings, or action vectors.  
- **Projection + damping** — reduces unsafe signal magnitude.  
- **Optional regularizer** — gentle nudging toward aligned behaviors.  
- **MIT‑licensed** — free for personal and commercial use.

---

## 📦 Install

```bash
git clone https://github.com/DerekEarnhart/safety-operator.git
cd safety-operator
pip install -e .
```

---

## 🛠 Quick Start

```python
import numpy as np
from safety_operator import SafetyOperator, SafetyConfig, SafeProjector, Regularizer, SafetyWrapper

# Example: mask certain logits as always‑safe
safe_mask = np.array([1, 1, 1, 0, 0], dtype=float)
proj = SafeProjector(safe_mask=safe_mask)

op = SafetyOperator(
    projector=proj,
    regularizer=Regularizer(l2_scale=0.05),
    config=SafetyConfig(alpha=0.8, beta=0.5, gamma=0.1),
)

wrapper = SafetyWrapper(operator=op, projector=proj)

logits = np.array([3.2, -1.0, 0.5, 8.0, -4.0])
filtered = wrapper.filter_logits(logits, steps=2)
print(filtered)
```

---

## 📊 Demo

<!-- Replace demo.png with your screenshot or one‑pager -->
![Safety Operator Demo](demo.png)

---

## 💄 Citation

If you use this in research or production, please cite:

```bibtex
@software{safety_operator,
  author    = {Earnhart, Derek},
  title     = {Safety Operator — Open Source AI "Seatbelt"},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.16790421},
  url       = {https://doi.org/10.5281/zenodo.16790421}
}
```

---

## 🔗 Links

- 📜 [Zenodo DOI](https://doi.org/10.5281/zenodo.16790421)  
- 🗜️ [MIT License](LICENSE)

---

## 👥 Contributing

Pull requests and suggestions are welcome.  
For major changes, please open an issue first to discuss what you’d like to change.

---

## 🌟 Support

If you find this useful, star the repo and share it.  
Your support helps more developers adopt safety mechanisms for AI.
