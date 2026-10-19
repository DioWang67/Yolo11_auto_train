# Dependency Management Guide

## 📦 Overview

This project uses `pip-tools` for reproducible dependency management.

## 📁 Dependency Files

- **`requirements.in`** - Core production dependencies (input)
- **`requirements.txt`** - Locked production dependencies (generated)
- **`requirements-dev.in`** - Development dependencies (input)
- **`requirements-dev.txt`** - Locked development dependencies (generated)

## 🚀 Installation

### For Production

```bash
pip install -r requirements.txt
```

### For Development

```bash
pip install -r requirements-dev.txt
```

This installs both production and development dependencies.

### For GUI Features

```bash
pip install -e ".[gui]"
```

## 🔧 Updating Dependencies

### Adding a New Dependency

1. Add to `requirements.in` (production) or `requirements-dev.in` (development)
2. Recompile lock files:

```bash
pip-compile requirements.in
pip-compile requirements-dev.in
```

3. Install updated dependencies:

```bash
pip install -r requirements-dev.txt
```

### Upgrading Dependencies

To upgrade all dependencies:

```bash
pip-compile --upgrade requirements.in
pip-compile --upgrade requirements-dev.in
```

To upgrade a specific package:

```bash
pip-compile --upgrade-package ultralytics requirements.in
```

## 📊 Dependency Groups

### Core Production
- **Deep Learning**: ultralytics, onnx, onnxruntime, onnxslim
- **Anomaly detection**: anomalib 2.7 (with lightning 2.6.6, kornia, timm, FrEIA)

The ML stack is pinned as one unit and only moves together, after a
station-validated upgrade: Python 3.11, torch 2.13.0 / torchvision 0.28.0,
lightning 2.6.6, anomalib 2.7.0, ultralytics 8.3.253, numpy 1.26.4,
opencv-python and opencv-python-headless 4.11.0.86, onnxruntime 1.23.2.
`picture-tool-doctor` checks exactly this set, and `.github/dependabot.yml`
keeps the bot from proposing pieces of it.

- **Computer Vision**: opencv-python, Pillow, albumentations
- **Data Science**: numpy, pandas, scipy, scikit-learn
- **CLI**: typer
- **Experiment Tracking**: mlflow (`tracking.enabled` in the pipeline configs)
- **Utilities**: tqdm, pyyaml

### Development Only
- **Testing**: pytest, pytest-cov, pytest-mock
- **Code Quality**: ruff, black, mypy, pylint
- **Documentation**: sphinx
- **Build**: pip-tools, build, setuptools

### Optional
- **GUI**: PyQt5 (install with `.[gui]`)
- **HTTP serve** (`picture_tool.serve`): fastapi, uvicorn, python-multipart
  (install with `.[serve]`; `requirements-dev.in` includes it for the tests)

## 🛡️ Best Practices

1. **Never edit `.txt` files directly** - Only edit `.in` files
2. **Always recompile after changes** - Run `pip-compile`
3. **Commit lock files** - Both `.in` and `.txt` files
4. **Test after updates** - Run tests after dependency updates
5. **Document breaking changes** - Note in CHANGELOG if dependency update breaks compatibility

## 🔍 Troubleshooting

### Dependency Conflicts

```bash
pip-compile --resolver=backtracking requirements.in
```

### Clean Install

```bash
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

### Verify Installation

```bash
pip check
```

## 📝 CI/CD Integration

### GitHub Actions Example

```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -r requirements-dev.txt
```

### Docker Example

```docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```
