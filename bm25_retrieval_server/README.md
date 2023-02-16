# Pyserini-based BM25 retrieval server

This is a simple BM25 retrieval server based on [Pyserini](https://github.com/castorini/pyserini).

## Usage

### 1. Set the environment variables

```bash
export $JAVA_HOME=/path/to/your/jdk_1_11
export LD_LIBRARY_PATH=$JAVA_HOME/lib/server:$LD_LIBRARY_PATH
export PYSERINI_CACHE=/path/to/your/pyserini_cache
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the server

```bash
python -u run_server.py
```
