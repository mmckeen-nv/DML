# DML MCP

This directory hosts the Model Context Protocol (MCP) server entrypoints for
the Daystrom Memory Lattice and CMA.

## Run the DML MCP server
```bash
pip install .[mcp]
dml-mcp-server --transport stdio
```

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
