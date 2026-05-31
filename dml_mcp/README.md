# DML MCP

This directory hosts the Model Context Protocol (MCP) server entrypoints for
the Daystrom Memory Lattice and CMA.

## Run the DML MCP server
```bash
pip install .[mcp]
dml-mcp-server --transport stdio
```

Tools:
- `search`: return memory result handles for provider-style retrieval
- `fetch`: return one memory by id
- `query`: query the lattice and receive structured context
- `ingest`: ingest local files/directories
- `stats`: return adapter stats

HTTP transport example:
```bash
dml-mcp-server --transport streamable-http --host 0.0.0.0 --port 8001
```

## Run the CMA MCP server
```bash
python -m dml_mcp.cma_mcp_server
```

## Docker
```bash
docker build -f dml_mcp/Dockerfile -t daystrom-dml-mcp .
docker run -p 8001:8001 daystrom-dml-mcp
```
