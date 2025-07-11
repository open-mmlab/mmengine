## Build Documentation

1. Clone MMEngine

   ```bash
   git clone https://github.com/vbti-development/onedl-mmengine.git
   cd mmengine
   ```

2. Install the building dependencies of documentation

   ```bash
   uv pip install -r pyproject.toml --group docs
   ```

3. Change directory to `docs/en`

   ```bash
   cd docs/en  # or docs/zh_cn
   ```

4. Build documentation

   ```bash
   make html
   ```

5. Open `_build/html/index.html` with browser
