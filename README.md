# Ollama-OpenAI Proxy

This is a Go-based proxy server that enables applications designed to work with the Ollama API to interact seamlessly with an OpenAI-compatible endpoint. It translates and forwards requests and responses between the two APIs while applying custom transformations to the model names and data formats.

**Note:** This is a pet project I use to forward requests to LiteLLM for use with Kerlig, which doesn’t support custom OpenAI endpoints. As this is a personal project, there might be issues and rough edges. Contributions and feedback are welcome!

## Features

**Endpoint Proxying:**
- **/v1/models & /v1/completions:** These endpoints are forwarded directly to the downstream OpenAI-compatible server.
- **/api/tags:** Queries the downstream `/v1/models` endpoint, transforms the response into the Ollama-style model list, and appends `:proxy` to model names if they don’t already contain a colon.
- **/api/chat:** Rewrites the request to the downstream `/v1/chat/completions` endpoint. It intercepts and transforms streaming NDJSON responses from the OpenAI format into the expected Ollama format, including stripping any trailing `:proxy` from model names.
- **/api/pull and other unknown endpoints:** Forwarded to a local Ollama instance running on `127.0.0.1:11505`.

**Debug Logging:**
When running in debug mode, the proxy logs:
- Every incoming request.
- The outgoing `/api/chat` payload.
- Raw downstream streaming chunks and their transformed equivalents.

**Model Name Handling:**
- For `/api/tags`, if a model ID does not contain a colon, the proxy appends `:proxy` to the name.
- For other endpoints, any `:proxy` suffix in model names is stripped before forwarding.

## Prerequisites

- **Go 1.18+** installed.
- An OpenAI-compatible server endpoint (e.g., running on `http://127.0.0.1:4000`).
- *(Optional)* A local Ollama instance running on `127.0.0.1:11505` for endpoints not handled by the downstream server.

## Installation

Clone this repository:

```
git clone https://github.com/yourusername/ollama-openai-proxy.git
cd ollama-openai-proxy
```

Build the project:

```
go build -o proxy-server main.go
```

## Usage

Run the proxy server with the desired flags:

```
./proxy-server --listen=":11434" --target="http://127.0.0.1:4000" --api-key="YOUR_API_KEY" --debug
```

## Command-Line Flags

- `--listen`: The address and port the proxy server listens on (default `:11434`).
- `--target`: The base URL of the OpenAI-compatible downstream server (e.g., `http://127.0.0.1:4000`).
- `--api-key`: *(Optional)* The API key for the downstream server.
- `--debug`: Enable detailed debug logging for every request and response.

## How It Works

1. **Request Routing:**
   The proxy intercepts requests and routes them based on the endpoint:
   - Requests to `/v1/models` and `/v1/completions` are forwarded directly.
   - Requests to `/api/tags` are handled locally by querying `/v1/models` on the downstream, transforming the JSON response, and appending `:proxy` where needed.
   - Requests to `/api/chat` are rewritten to `/v1/chat/completions`, with the payload and response processed to strip or add the `:proxy` suffix as appropriate.
   - All other endpoints are forwarded to the local Ollama instance.

2. **Response Transformation:**
   Streaming responses from the downstream `/v1/chat/completions` endpoint (in NDJSON format) are read line by line. Each chunk is parsed, transformed into the Ollama format, and streamed back to the client.

3. **Logging:**
   With debug mode enabled, detailed logs of incoming requests, outgoing payloads, and both raw and transformed response chunks are printed.

## Contributing

Contributions are welcome! As this is a pet project, there may be rough edges and issues. Please feel free to open issues or submit pull requests for improvements and bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
