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
	"path"
	"strings"
	"time"
)

// --------------------
// Data Structures
// --------------------

type DownstreamModel struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
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
	ModifiedAt string       `json:"modified_at"`
	Size       int64        `json:"size"`
	Digest     string       `json:"digest"`
	Details    ModelDetails `json:"details"`
}

// PSModel extends Model struct with additional fields needed for the /api/ps endpoint
type PSModel struct {
	Name      string       `json:"name"`
	Model     string       `json:"model"`
	Size      int64        `json:"size"`
	Digest    string       `json:"digest"`
	Details   ModelDetails `json:"details"`
	ExpiresAt string       `json:"expires_at"`
	SizeVram  int64        `json:"size_vram"`
}

type TagsResponse struct {
	Models []Model `json:"models"`
}

type PSResponse struct {
	Models []PSModel `json:"models"`
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

	// Create a reverse proxy for /models and /completions.
	proxy := httputil.NewSingleHostReverseProxy(targetUrl)
	originalDirector := proxy.Director
	proxy.Director = func(req *http.Request) {
		originalDirector(req)
		// For downstream endpoints, also set the API key if provided.
		if *openaiApiKey != "" {
			req.Header.Set("Authorization", "Bearer "+*openaiApiKey)
		}
	}

	// Handler for /models.
	http.HandleFunc("/models", func(w http.ResponseWriter, r *http.Request) {
		log.Printf("Proxying /models request to %s", targetUrl.String())
		proxy.ServeHTTP(w, r)
	})
	// Handler for /models.
	http.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		r.URL.Path = "/models"
		log.Printf("Proxying /models request to %s", targetUrl.String())
		proxy.ServeHTTP(w, r)
	})

	// Handler for /completions.
	http.HandleFunc("/completions", func(w http.ResponseWriter, r *http.Request) {
		log.Printf("Proxying /completions request to %s", targetUrl.String())
		proxy.ServeHTTP(w, r)
	})
	// Handler for /completions.
	http.HandleFunc("/v1/completions", func(w http.ResponseWriter, r *http.Request) {
		r.URL.Path = "/completions"
		log.Printf("Proxying /completions request to %s", targetUrl.String())
		proxy.ServeHTTP(w, r)
	})
	// Handler for /completions.
	http.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		r.URL.Path = "/chat/completions"
		log.Printf("Proxying /completions request to %s", targetUrl.String())
		proxy.ServeHTTP(w, r)
	})

	// Handler for /api/tags.
	http.HandleFunc("/api/tags", func(w http.ResponseWriter, r *http.Request) {
		log.Printf("Handling /api/tags request by querying downstream /models")
		modelsURL := *targetUrl
		modelsURL.Path = path.Join(targetUrl.Path, "models")
		reqDown, err := http.NewRequest("GET", modelsURL.String(), nil)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if *openaiApiKey != "" {
			reqDown.Header.Set("Authorization", "Bearer "+*openaiApiKey)
		}
		if *debug {
			if dump, err := httputil.DumpRequestOut(reqDown, true); err == nil {
				log.Printf("Outgoing /models request:\n%s", dump)
			} else {
				log.Printf("Error dumping /models request: %v", err)
			}
		}
		client := &http.Client{}
		respDown, err := client.Do(reqDown)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		defer respDown.Body.Close()
		if *debug {
			if dump, err := httputil.DumpResponse(respDown, false); err == nil {
				log.Printf("Received response from /models:\n%s", dump)
			} else {
				log.Printf("Error dumping /models response: %v", err)
			}
		}
		var dsResp DownstreamModelsResponse
		if err := json.NewDecoder(respDown.Body).Decode(&dsResp); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		
		// Properly formatted timestamp for Ollama
		timeStr := time.Now().UTC().Format(time.RFC3339Nano)
		
		var tagsResp TagsResponse
		for _, dm := range dsResp.Data {
			modelName := dm.ID
			if !strings.Contains(modelName, ":") {
				modelName += ":proxy"
			}
			
			// Create a placeholder hash as digest
			digest := "9f438cb9cd581fc025612d27f7c1a6669ff83a8bb0ed86c94fcf4c5440555697"
			
			modelEntry := Model{
				Name:       modelName,
				ModifiedAt: timeStr,
				Size:       3825819519, // Placeholder size
				Digest:     digest,
				Details: ModelDetails{
					Format:            "gguf",
					Family:            "llama",
					Families:          nil,
					ParameterSize:     "7B",
					QuantizationLevel: "Q4_0",
				},
			}
			tagsResp.Models = append(tagsResp.Models, modelEntry)
		}
		
		// If no models were found, ensure we return at least an empty array
		if tagsResp.Models == nil {
			tagsResp.Models = []Model{}
		}
		
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(tagsResp); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	})
	
	// Handler for /api/ps to list running models
	http.HandleFunc("/api/ps", func(w http.ResponseWriter, r *http.Request) {
		log.Printf("Handling /api/ps request by querying downstream /models")
		modelsURL := *targetUrl
		modelsURL.Path = path.Join(targetUrl.Path, "models")
		reqDown, err := http.NewRequest("GET", modelsURL.String(), nil)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if *openaiApiKey != "" {
			reqDown.Header.Set("Authorization", "Bearer "+*openaiApiKey)
		}
		if *debug {
			if dump, err := httputil.DumpRequestOut(reqDown, true); err == nil {
				log.Printf("Outgoing /models request for /api/ps:\n%s", dump)
			} else {
				log.Printf("Error dumping /models request: %v", err)
			}
		}
		client := &http.Client{}
		respDown, err := client.Do(reqDown)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		defer respDown.Body.Close()
		if *debug {
			if dump, err := httputil.DumpResponse(respDown, false); err == nil {
				log.Printf("Received response from /models for /api/ps:\n%s", dump)
			} else {
				log.Printf("Error dumping /models response: %v", err)
			}
		}
		var dsResp DownstreamModelsResponse
		if err := json.NewDecoder(respDown.Body).Decode(&dsResp); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		
		// Create expiry time (24 hours from now)
		expiryTime := time.Now().Add(24 * time.Hour).Format(time.RFC3339Nano)
		
		var psResp PSResponse
		for _, dm := range dsResp.Data {
			modelName := dm.ID
			if !strings.Contains(modelName, ":") {
				modelName += ":proxy"
			}
			
			// Create a placeholder hash as digest
			digest := "2ae6f6dd7a3dd734790bbbf58b8909a606e0e7e97e94b7604e0aa7ae4490e6d8"
			
			// Create families array
			families := []string{"llama"}
			
			modelEntry := PSModel{
				Name:   modelName,
				Model:  modelName,
				Size:   5137025024, // Placeholder size
				Digest: digest,
				Details: ModelDetails{
					ParentModel:       "",
					Format:            "gguf",
					Family:            "llama",
					Families:          families,
					ParameterSize:     "7.2B",
					QuantizationLevel: "Q4_0",
				},
				ExpiresAt: expiryTime,
				SizeVram:  5137025024,
			}
			psResp.Models = append(psResp.Models, modelEntry)
		}
		
		// If no models were found, ensure we return at least an empty array
		if psResp.Models == nil {
			psResp.Models = []PSModel{}
		}
		
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(psResp); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	})

	// Explicit handler for /api/pull: return 404 instead of forwarding.
	http.HandleFunc("/api/pull", func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "Endpoint /api/pull is not supported", http.StatusNotFound)
	})

	// Handler for /api/show to return model information.
	http.HandleFunc("/api/show", func(w http.ResponseWriter, r *http.Request) {
		log.Println("Handling /api/show request")

		// Parse the model name from the query parameters
		modelName := r.URL.Query().Get("model")
		if modelName == "" {
                        modelName = "LLAMA"
		}

		// Strip :proxy suffix if present
		modelName = strings.TrimSuffix(modelName, ":proxy")

		// Create response structure
		modelInfo := map[string]interface{}{
			"modelfile": "# Modelfile generated by \"ollama show\"\n# To build a new Modelfile based on this one, replace the FROM line with:\n# FROM " + modelName + ":latest\n\nFROM " + modelName + "\nTEMPLATE \"\"\"{{ .System }}\nUSER: {{ .Prompt }}\nASSISTANT: \"\"\"",
			"parameters": "num_keep                       24\nstop                           \"<|start_header_id|>\"\nstop                           \"<|end_header_id|>\"\nstop                           \"<|eot_id|>\"",
			"template": "{{ if .System }}<|start_header_id|>system<|end_header_id|>\n\n{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>\n\n{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>\n\n{{ .Response }}<|eot_id|>",
			"details": map[string]interface{}{
				"parent_model":      "",
				"format":            "gguf",
				"family":            "llama",
				"families":          []string{"llama"},
				"parameter_size":    "8.0B",
				"quantization_level": "Q4_0",
			},
			"model_info": map[string]interface{}{
				"general.architecture":              "llama",
				"general.file_type":                 2,
				"general.parameter_count":           8030261248,
				"general.quantization_version":      2,
				"llama.attention.head_count":        32,
				"llama.attention.head_count_kv":     8,
				"llama.attention.layer_norm_rms_epsilon": 0.00001,
				"llama.block_count":                 32,
				"llama.context_length":              8192,
				"llama.embedding_length":            4096,
				"llama.feed_forward_length":         14336,
				"llama.rope.dimension_count":        128,
				"llama.rope.freq_base":              500000,
				"llama.vocab_size":                  128256,
				"tokenizer.ggml.bos_token_id":       128000,
				"tokenizer.ggml.eos_token_id":       128009,
				"tokenizer.ggml.merges":             []string{},
				"tokenizer.ggml.model":              "gpt2",
				"tokenizer.ggml.pre":                "llama-bpe",
				"tokenizer.ggml.token_type":         []string{},
				"tokenizer.ggml.tokens":             []string{},
			},
		}
		
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(modelInfo); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	})

	// Explicit handler for /api/chat.
	// This handler rewrites the URL to /chat/completions, logs the outgoing payload,
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

		// Unmarshal and modify the request payload: strip ":proxy" from model field
		// and remove unsupported options.
		var payload map[string]interface{}
		if err := json.Unmarshal(bodyBytes, &payload); err == nil {
			// Remove unsupported "options" field.
			delete(payload, "options")
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

		// Create a new request with joined path: /v2/ai/chat/completions.
		newURL := *targetUrl
		newURL.Path = path.Join(targetUrl.Path, "chat/completions")
		newReq, err := http.NewRequest("POST", newURL.String(), bytes.NewReader(bodyBytes))
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		newReq.Header = r.Header.Clone()
		if *openaiApiKey != "" {
			newReq.Header.Set("Authorization", "Bearer "+*openaiApiKey)
		}

		// Log the full outgoing /api/chat request.
		if *debug {
			if dump, err := httputil.DumpRequestOut(newReq, true); err == nil {
				log.Printf("Outgoing /api/chat request:\n%s", dump)
			} else {
				log.Printf("Error dumping /api/chat request: %v", err)
			}
		}

		client := &http.Client{}
		resp, err := client.Do(newReq)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}

		// Log the response headers (without draining the body).
		if *debug {
			if dump, err := httputil.DumpResponse(resp, false); err == nil {
				log.Printf("Received response from /chat/completions:\n%s", dump)
			} else {
				log.Printf("Error dumping /chat/completions response: %v", err)
			}
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
				CreatedAt: time.Now().Format(time.RFC3339),
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
	
	// OllamaGenerateChunk represents the output format for the /api/generate endpoint.
	type OllamaGenerateChunk struct {
		Model     string `json:"model"`
		CreatedAt string `json:"created_at"`
		Response  string `json:"response"`
		Done      bool   `json:"done"`
	}
	
	// Handler for /api/generate
	// This handler rewrites the URL to /completions, logs the outgoing payload,
	// strips any trailing ":proxy" from the model name in the request payload,
	// intercepts the downstream streaming response, transforms each chunk from OpenAI format
	// to Ollama format, and streams the transformed chunks to the client.
	http.HandleFunc("/api/generate", func(w http.ResponseWriter, r *http.Request) {
		log.Println("Handling /api/generate transformation")
		// Read the original request body.
		bodyBytes, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		r.Body.Close()
		if *debug {
			log.Printf("Outgoing /api/generate payload: %s", string(bodyBytes))
		}

		// Unmarshal and modify the request payload: strip ":proxy" from model field
		// and transform to OpenAI completions format
		var payload map[string]interface{}
		if err := json.Unmarshal(bodyBytes, &payload); err == nil {
			// Remove unsupported fields
			delete(payload, "options")
			
			// Extract model and prompt
			var model string
			if modelVal, ok := payload["model"].(string); ok {
				model = strings.TrimSuffix(modelVal, ":proxy")
			}
			
			var prompt string
			if promptVal, ok := payload["prompt"].(string); ok {
				prompt = promptVal
			}
			
			// Create a new payload in OpenAI completions format
			openaiPayload := map[string]interface{}{
				"model": model,
				"prompt": prompt,
				"stream": true,
				"max_tokens": 2048, // Default value, can be configurable
			}
			
			// Re-marshal payload to OpenAI format
			bodyBytes, err = json.Marshal(openaiPayload)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
		} else {
			log.Printf("Warning: could not unmarshal payload for transformation: %v", err)
		}

		// Create a new request with joined path to /completions
		newURL := *targetUrl
		newURL.Path = path.Join(targetUrl.Path, "completions")
		newReq, err := http.NewRequest("POST", newURL.String(), bytes.NewReader(bodyBytes))
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		newReq.Header = r.Header.Clone()
		if *openaiApiKey != "" {
			newReq.Header.Set("Authorization", "Bearer "+*openaiApiKey)
		}

		// Log the full outgoing request
		if *debug {
			if dump, err := httputil.DumpRequestOut(newReq, true); err == nil {
				log.Printf("Outgoing /completions request:\n%s", dump)
			} else {
				log.Printf("Error dumping /completions request: %v", err)
			}
		}

		client := &http.Client{}
		resp, err := client.Do(newReq)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}

		// Log the response headers (without draining the body)
		if *debug {
			if dump, err := httputil.DumpResponse(resp, false); err == nil {
				log.Printf("Received response from /completions:\n%s", dump)
			} else {
				log.Printf("Error dumping /completions response: %v", err)
			}
		}

		defer resp.Body.Close()
		// Copy response headers
		for key, values := range resp.Header {
			for _, value := range values {
				w.Header().Add(key, value)
			}
		}
		w.WriteHeader(resp.StatusCode)

		// Process streaming NDJSON response
		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if *debug {
				log.Printf("Raw downstream chunk: %s", line)
			}
			// Strip off the SSE "data:" prefix if present
			if strings.HasPrefix(line, "data:") {
				line = strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			}
			// Skip if the line is empty or indicates completion
			if line == "" || line == "[DONE]" {
				continue
			}
			
			// Parse the JSON chunk from OpenAI completions format
			var openaiChunk struct {
				ID      string `json:"id"`
				Object  string `json:"object"`
				Created int64  `json:"created"`
				Model   string `json:"model"`
				Choices []struct {
					Text         string  `json:"text"`
					Index        int     `json:"index"`
					FinishReason *string `json:"finish_reason,omitempty"`
				} `json:"choices"`
			}
			
			if err := json.Unmarshal([]byte(line), &openaiChunk); err != nil {
				log.Printf("Error unmarshalling chunk: %v", err)
				// In case of error, send the raw line
				w.Write([]byte(line + "\n"))
				continue
			}
			
			// Transform the chunk into Ollama generate format
			var text string
			done := false
			if len(openaiChunk.Choices) > 0 {
				choice := openaiChunk.Choices[0]
				text = choice.Text
				if choice.FinishReason != nil && *choice.FinishReason != "" {
					done = true
				}
			}
			
			// Strip any ":proxy" from the model name
			modelName := strings.TrimSuffix(openaiChunk.Model, ":proxy")
			transformed := OllamaGenerateChunk{
				Model:     modelName,
				CreatedAt: time.Now().Format(time.RFC3339),
				Response:  text,
				Done:      done,
			}
			
			transformedLine, err := json.Marshal(transformed)
			if err != nil {
				log.Printf("Error marshalling transformed chunk: %v", err)
				w.Write([]byte(line + "\n"))
				continue
			}
			
			if *debug {
				log.Printf("Transformed generate chunk: %s", string(transformedLine))
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
			w.Header().Set("Content-Type", "text/plain")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("Ollama is running"))
			return
		}
	})

	log.Printf("Proxy server listening on %s\n- /models & /completions forwarded to %s\n- /api/tags & /api/ps dynamically transformed\n- /api/pull returns 404\n- /api/chat rewritten and transformed before forwarding to downstream (/chat/completions)\n- Unknown endpoints will%s be forwarded to 127.0.0.1:11505",
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
