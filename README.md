# README.md
password
wqu


If you're working with a notebook that relies on pip (e.g., via the %pip magic), you can include pip in your project's virtual environment by running uv venv --seed prior to starting the Jupyter server. For example, given:


uv venv --seed
uv run --with jupyter jupyter lab

https://docs.astral.sh/uv/guides/integration/jupyter/#installing-packages-without-a-kernel