package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"time"
)

// --------------------
// Data Structures
// --------------------

// Structures used for /api/tags transformation.
type DownstreamModel struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type DownstreamModelsResponse struct {
	Data   []DownstreamModel `json:"data"`
	Object string            `json:"object"`
}

type ModelDetails struct {
	ParentModel       string   `json:"parent_model"`
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
}

type Model struct {
	Name       string       `json:"name"`
	Model      string       `json:"model"`
	ModifiedAt string       `json:"modified_at"`
	Size       int64        `json:"size"`
	Digest     string       `json:"digest"`
	Details    ModelDetails `json:"details"`
}

type TagsResponse struct {
	Models []Model `json:"models"`
}

// Structures used for transforming the /api/chat response.

// OpenAIChunk represents one NDJSON chunk from the OpenAI‑compatible streaming endpoint.
type OpenAIChunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Delta struct {
			Role    string `json:"role,omitempty"`
			Content string `json:"content,omitempty"`
		} `json:"delta"`
		Index        int     `json:"index"`
		FinishReason *string `json:"finish_reason,omitempty"`
	} `json:"choices"`
}

// OllamaChunk is the expected response format for Ollama.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OllamaChunk struct {
	Model     string  `json:"model"`
	CreatedAt string  `json:"created_at"`
	Message   Message `json:"message"`
	Done      bool    `json:"done"`
}

// --------------------
// Helpers & Middleware
// --------------------

// logMiddleware logs every HTTP request when debug mode is enabled.
func logMiddleware(debug bool, next http.Handler) http.Handler {
	if !debug {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		log.Printf("DEBUG: Received %s request for %s", r.Method, r.URL.String())
		next.ServeHTTP(w, r)
	})
}

// forwardToOllama logs the request and response before proxying the request to the local Ollama instance.
func forwardToOllama(w http.ResponseWriter, r *http.Request) {
	log.Println("=== Unknown Request Received ===")
	log.Printf("Method: %s", r.Method)
	log.Printf("URL: %s", r.URL.String())
	log.Printf("Headers: %v", r.Header)
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("Error reading request body: %v", err)
	} else if len(bodyBytes) > 0 {
		log.Printf("Body: %s", string(bodyBytes))
	} else {
		log.Printf("Body: <empty>")
	}
	r.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))

	targetOllama, err := url.Parse("http://127.0.0.1:11505")
	if err != nil {
		log.Printf("Error parsing target Ollama URL: %v", err)
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
		return
	}

	ollamaProxy := httputil.NewSingleHostReverseProxy(targetOllama)
	ollamaProxy.ModifyResponse = func(resp *http.Response) error {
		respBody, err := io.ReadAll(resp.Body)
		if err != nil {
			log.Printf("Error reading response body: %v", err)
			return err
		}
		log.Println("=== Response from 127.0.0.1:11505 ===")
		log.Printf("Status: %s", resp.Status)
		log.Printf("Headers: %v", resp.Header)
		log.Printf("Body: %s", string(respBody))
		resp.Body = io.NopCloser(bytes.NewBuffer(respBody))
		return nil
	}
	ollamaProxy.ServeHTTP(w, r)
}

// --------------------
// Main
// --------------------

func main() {
	// Command-line flags.
	listenAddr := flag.String("listen", ":11434", "Address to listen on (e.g. :11434)")
	targetUrlStr := flag.String("target", "http://127.0.0.1:4000", "Target OpenAI-compatible server URL")
	openaiApiKey := flag.String("api-key", "", "OpenAI API key (optional)")
	debug := flag.Bool("debug", false, "Print debug logs for every call")
	forwardUnknown := flag.Bool("forward-unknown", false, "Forward unknown endpoints to local Ollama instance at 127.0.0.1:11505")
	flag.Parse()

	// Parse the target URL.
	targetUrl, err := url.Parse(*targetUrlStr)
	if err != nil {
		log.Fatalf("Error parsing target URL: %v", err)
	}

	// Create a reverse proxy for /v1/models and /v1/completions.
	proxy := httputil.NewSingleHostReverseProxy(targetUrl)
	originalDirector := proxy.Director
	proxy.Director = func(req *http.Request) {
		originalDirector(req)
		// For downstream endpoints, also set the API key if provided.
		if *openaiApiKey != "" {
			req.Header.Set("Authorization", "Bearer "+*openaiApiKey)
		}
	}

	// Handler for /v1/models.
	http.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		log.Printf("Proxying /v1/models request to %s", targetUrl.String())
		proxy.ServeHTTP(w, r)
	})

	// Handler for /v1/completions.
	http.HandleFunc("/v1/completions", func(w http.ResponseWriter, r *http.Request) {
		log.Printf("Proxying /v1/completions request to %s", targetUrl.String())
		proxy.ServeHTTP(w, r)
	})

	// Handler for /api/tags.
	// When building the list, if a model's ID does not contain a colon,
	// append ":proxy" to it.
	http.HandleFunc("/api/tags", func(w http.ResponseWriter, r *http.Request) {
		log.Printf("Handling /api/tags request by querying downstream /v1/models")
		modelsURL := targetUrl.ResolveReference(&url.URL{Path: "/v1/models"})
		reqDown, err := http.NewRequest("GET", modelsURL.String(), nil)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if *openaiApiKey != "" {
			reqDown.Header.Set("Authorization", "Bearer "+*openaiApiKey)
		}
		client := &http.Client{}
		respDown, err := client.Do(reqDown)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		defer respDown.Body.Close()

		if respDown.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(respDown.Body)
			http.Error(w, string(body), respDown.StatusCode)
			return
		}

		var dsResp DownstreamModelsResponse
		if err := json.NewDecoder(respDown.Body).Decode(&dsResp); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		var tagsResp TagsResponse
		for _, dm := range dsResp.Data {
			modelName := dm.ID
			// Append ":proxy" if there is no colon in the model name.
			if !strings.Contains(modelName, ":") {
				modelName += ":proxy"
			}
			modelEntry := Model{
				Name:       modelName,
				Model:      modelName,
				ModifiedAt: time.Unix(dm.Created, 0).UTC().Format(time.RFC3339Nano),
				Size:       0,
				Digest:     "",
				Details: ModelDetails{
					ParentModel:       "",
					Format:            "unknown",
					Family:            "",
					Families:          []string{},
					ParameterSize:     "",
					QuantizationLevel: "",
				},
			}
			tagsResp.Models = append(tagsResp.Models, modelEntry)
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(tagsResp); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	})

	// Explicit handler for /api/pull: return 404 instead of forwarding.
	http.HandleFunc("/api/pull", func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "Endpoint /api/pull is not supported", http.StatusNotFound)
	})

	// Explicit handler for /api/chat.
	// This handler rewrites the URL to /v1/chat/completions, logs the outgoing payload,
	// strips any trailing ":proxy" from the model name in the request payload,
	// intercepts the downstream streaming response, transforms each chunk from OpenAI format
	// to Ollama format (stripping any ":proxy" from the model field), logs both the raw and transformed
	// chunks when debug is enabled, and streams the transformed chunks to the client.
	http.HandleFunc("/api/chat", func(w http.ResponseWriter, r *http.Request) {
		log.Println("Handling /api/chat transformation")
		// Read the original request body.
		bodyBytes, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		r.Body.Close()
		if *debug {
			log.Printf("Outgoing /api/chat payload: %s", string(bodyBytes))
		}

		// Unmarshal and modify the request payload: strip ":proxy" from model field.
		var payload map[string]interface{}
		if err := json.Unmarshal(bodyBytes, &payload); err == nil {
			if modelVal, ok := payload["model"].(string); ok {
				payload["model"] = strings.TrimSuffix(modelVal, ":proxy")
			}
			// Re-marshal payload.
			bodyBytes, err = json.Marshal(payload)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
		} else {
			log.Printf("Warning: could not unmarshal payload for transformation: %v", err)
		}

		// Create a new request to the downstream /v1/chat/completions endpoint.
		newURL := targetUrl.ResolveReference(&url.URL{Path: "/v1/chat/completions"})
		newReq, err := http.NewRequest("POST", newURL.String(), bytes.NewReader(bodyBytes))
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		newReq.Header = r.Header.Clone()
		if *openaiApiKey != "" {
			newReq.Header.Set("Authorization", "Bearer "+*openaiApiKey)
		}

		client := &http.Client{}
		resp, err := client.Do(newReq)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()

		// Copy response headers.
		for key, values := range resp.Header {
			for _, value := range values {
				w.Header().Add(key, value)
			}
		}
		w.WriteHeader(resp.StatusCode)

		// Process streaming NDJSON response.
		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if *debug {
				log.Printf("Raw downstream chunk: %s", line)
			}
			// Strip off the SSE "data:" prefix if present.
			if strings.HasPrefix(line, "data:") {
				line = strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			}
			// Skip if the line is empty or indicates completion.
			if line == "" || line == "[DONE]" {
				continue
			}
			// Parse the JSON chunk.
			var chunk OpenAIChunk
			if err := json.Unmarshal([]byte(line), &chunk); err != nil {
				log.Printf("Error unmarshalling chunk: %v", err)
				// In case of error, send the raw line.
				w.Write([]byte(line + "\n"))
				continue
			}
			// Transform the chunk into Ollama format.
			var content string
			role := "assistant" // default role
			done := false
			if len(chunk.Choices) > 0 {
				choice := chunk.Choices[0]
				content = choice.Delta.Content
				if choice.Delta.Role != "" {
					role = choice.Delta.Role
				}
				if choice.FinishReason != nil && *choice.FinishReason != "" {
					done = true
				}
			}
			// Strip any ":proxy" from the model name.
			modelName := strings.TrimSuffix(chunk.Model, ":proxy")
			transformed := OllamaChunk{
				Model:     modelName,
				CreatedAt: time.Unix(chunk.Created, 0).UTC().Format(time.RFC3339Nano),
				Message: Message{
					Role:    role,
					Content: content,
				},
				Done: done,
			}
			transformedLine, err := json.Marshal(transformed)
			if err != nil {
				log.Printf("Error marshalling transformed chunk: %v", err)
				w.Write([]byte(line + "\n"))
				continue
			}
			if *debug {
				log.Printf("Transformed chunk: %s", string(transformedLine))
			}
			w.Write(transformedLine)
			w.Write([]byte("\n"))
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
		if err := scanner.Err(); err != nil {
			log.Printf("Scanner error: %v", err)
		}
	})

	// Catch-all handler for any other unknown endpoints.
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if *forwardUnknown {
			forwardToOllama(w, r)
		} else {
			http.NotFound(w, r)
		}
	})

	log.Printf("Proxy server listening on %s\n- /v1/models & /v1/completions forwarded to %s\n- /api/tags dynamically transformed\n- /api/pull returns 404\n- /api/chat rewritten and transformed before forwarding to downstream (/v1/chat/completions)\n- Unknown endpoints will%s be forwarded to 127.0.0.1:11505",
		*listenAddr, targetUrl.String(), func() string {
			if *forwardUnknown {
				return ""
			}
			return " NOT"
		}())
	if err := http.ListenAndServe(*listenAddr, logMiddleware(*debug, http.DefaultServeMux)); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
