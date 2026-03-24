# Aerospike LangGraph

Aerospike-backed persistence for [LangGraph](https://github.com/langchain-ai/langgraph). This monorepo provides drop-in checkpoint and store implementations so your LangGraph agents can durably save state to an [Aerospike](https://aerospike.com/) cluster.

## Packages

| Package                                                                     | Description                                          | Install                                         |
| --------------------------------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------- |
| [langgraph-checkpoint-aerospike](./packages/langgraph-checkpoint-aerospike) | Checkpoint saver for LangGraph graph execution state | `pip install -U langgraph-checkpoint-aerospike` |
| [langgraph-store-aerospike](./packages/langgraph-store-aerospike)           | Key/value store with batch ops, search, and TTL      | `pip install -U langgraph-store-aerospike`      |

## Requirements

- Python >= 3.10
- Aerospike Server (or the [Docker image](https://hub.docker.com/_/aerospike))
- `aerospike` Python client >= 15
- `langgraph` >= 0.6

## Quick Start

### 1. Start Aerospike

```bash
docker run -d --name aerospike -p 3000-3002:3000-3002 container.aerospike.com/aerospike/aerospike-server
```

### 2. Install

```bash
pip install -U langgraph-checkpoint-aerospike langgraph-store-aerospike
```

### 3. Use the Checkpoint Saver

```python
import aerospike
from langgraph.checkpoint.aerospike import AerospikeSaver

client = aerospike.client({"hosts": [("127.0.0.1", 3000)]}).connect()
saver = AerospikeSaver(client=client, namespace="test")

compiled = graph.compile(checkpointer=saver)
compiled.invoke({"input": "hello"}, config={"configurable": {"thread_id": "demo"}})
```

### 4. Use the Store

```python
import aerospike
from langgraph.store.aerospike import AerospikeStore

client = aerospike.client({"hosts": [("127.0.0.1", 3000)]}).connect()
store = AerospikeStore(client=client, namespace="test", set="langgraph_store")

store.put(namespace=("users", "profiles"), key="user_123", value={"name": "Alice", "age": 30})
item = store.get(namespace=("users", "profiles"), key="user_123")
```

## Configuration

Both packages read connection details from environment variables by default:

| Variable              | Default              | Description                 |
| --------------------- | -------------------- | --------------------------- |
| `AEROSPIKE_HOST`      | `127.0.0.1`          | Aerospike cluster seed host |
| `AEROSPIKE_PORT`      | `3000`               | Aerospike cluster seed port |
| `AEROSPIKE_NAMESPACE` | `test`               | Aerospike namespace to use  |
| `AEROSPIKE_SET`       | _(package-specific)_ | Aerospike set name          |

## Development

```bash
# Clone the repo
git clone https://github.com/aerospike/aerospike-langgraph.git
cd aerospike-langgraph

# Install a package in editable mode
pip install -e packages/langgraph-checkpoint-aerospike
pip install -e packages/langgraph-store-aerospike

# Run tests (requires a running Aerospike instance)
pytest packages/langgraph-checkpoint-aerospike/tests
pytest packages/langgraph-store-aerospike/tests
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) for linting, commit conventions, and CI details.

## License

[Apache 2.0](./LICENSE)
