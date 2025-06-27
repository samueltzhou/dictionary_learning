```bash
curl -sSL https://install.python-poetry.org | python3.10 -;
poetry config virtualenvs.in-project true
cd /home/samtzhou/dictionary_learning    # adjust if necessary
poetry env use python3.10
poetry install
poetry env activate
```