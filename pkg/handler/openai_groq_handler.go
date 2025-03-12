package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"github.com/gin-gonic/gin"
	"github.com/sashabaranov/go-openai"
	"go.uber.org/zap"
	"io"
	"net/http"
	"simple-one-api/pkg/mylog"
)

// https://console.groq.com/docs/openai
func adjustGroqReq(req *openai.ChatCompletionRequest) {
	req.LogProbs = false
	req.LogitBias = nil
	req.TopLogProbs = 0
	if req.N != 0 {
		req.N = 1
	}

	if req.Temperature <= 0 {
		req.Temperature = 0.1
	}

	if req.Temperature > 2 {
		req.Temperature = 2
	}
}

// Custom Groq request structure to hold additional parameters
type GroqCustomRequest struct {
	*openai.ChatCompletionRequest
	ReasoningFormat string `json:"reasoning_format,omitempty"`
}

// getGroqReasoningFormat determines the appropriate reasoning_format value
// based on the request context and configuration
func getGroqReasoningFormat(req *openai.ChatCompletionRequest, credentials map[string]interface{}) string {
	// First check if reasoning_format is explicitly specified in credentials
	if reasoningFormat, exists := credentials["reasoning_format"]; exists {
		if formatStr, ok := reasoningFormat.(string); ok {
			// Validate the value is one of the allowed values
			switch formatStr {
			case "raw", "parsed", "hidden":
				return formatStr
			default:
				mylog.Logger.Warn("Invalid reasoning_format specified in credentials", 
					zap.String("value", formatStr), 
					zap.String("using_default", "raw"))
			}
		}
	}
	
	// Default behavior: if JSON mode is enabled, use "parsed"
	if req.ResponseFormat != nil && req.ResponseFormat.Type == openai.ChatCompletionResponseFormatTypeJSONObject {
		return "parsed"
	}
	
	// If tools are being used, also default to "parsed"
	if len(req.Tools) > 0 {
		return "parsed"
	}
	
	// Otherwise default to "raw"
	return "raw"
}

// GroqTransport is a custom HTTP transport that adds reasoning_format to Groq requests
type GroqTransport struct {
	Transport      http.RoundTripper
	ReasoningFormat string
}

// RoundTrip implements the http.RoundTripper interface
func (g *GroqTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Only modify chat completion requests to Groq
	if req.URL.Path == "/v1/chat/completions" {
		// Read the original request body
		var body []byte
		var err error
		if req.Body != nil {
			body, err = io.ReadAll(req.Body)
			req.Body.Close()
			if err != nil {
				return nil, err
			}
		}

		// Parse the request
		var oaiReq openai.ChatCompletionRequest
		if err := json.Unmarshal(body, &oaiReq); err != nil {
			return nil, err
		}

		// Create a custom Groq request with the reasoning_format parameter
		groqReq := GroqCustomRequest{
			ChatCompletionRequest: &oaiReq,
			ReasoningFormat:       g.ReasoningFormat,
		}

		// Marshal back to JSON
		modifiedBody, err := json.Marshal(groqReq)
		if err != nil {
			return nil, err
		}

		// Log the modified request
		mylog.Logger.Debug("Modified Groq request with reasoning_format",
			zap.String("reasoning_format", groqReq.ReasoningFormat),
			zap.String("body", string(modifiedBody)))

		// Set the new body
		req.Body = io.NopCloser(bytes.NewReader(modifiedBody))
		req.ContentLength = int64(len(modifiedBody))
	}

	// Pass the (potentially modified) request to the underlying transport
	return g.Transport.RoundTrip(req)
}

// OpenAI2GroqOpenAIHandler handles OpenAI to Groq OpenAI requests
func OpenAI2GroqOpenAIHandler(c *gin.Context, oaiReqParam *OAIRequestParam) error {
	req := oaiReqParam.chatCompletionReq
	s := oaiReqParam.modelDetails
	credentials := oaiReqParam.creds
	
	conf, err := getConfig(s, oaiReqParam)
	if err != nil {
		return err
	}

	adjustGroqReq(req)
	
	// Determine the reasoning_format parameter from credentials or defaults
	reasoningFormat := getGroqReasoningFormat(req, credentials)

	// Use our custom GroqTransport with the existing transport
	defaultTransport := http.DefaultTransport
	if oaiReqParam.httpTransport != nil {
		defaultTransport = oaiReqParam.httpTransport
	}
	
	groqTransport := &GroqTransport{
		Transport: defaultTransport,
		ReasoningFormat: reasoningFormat,
	}
	
	// Use the custom transport in the HTTP client
	conf.HTTPClient = &http.Client{
		Transport: groqTransport,
	}

	mylog.Logger.Info("Using Groq API with reasoning_format", 
		zap.String("reasoning_format", reasoningFormat),
		zap.String("model", req.Model))

	clientModel := oaiReqParam.ClientModel
	// Use the standard OpenAI handler with our custom config
	return handleOpenAIOpenAIRequest(conf, c, req, clientModel)
}

// handleGroqOpenAIRequest is a specialized handler for Groq requests
// that adds the reasoning_format parameter
func handleGroqOpenAIRequest(conf openai.ClientConfig, c *gin.Context, req *openai.ChatCompletionRequest, clientModel string) error {
	logOpenAIChatCompletionRequest(req)

	openaiClient := openai.NewClientWithConfig(conf)
	ctx := context.Background()

	// For Groq, we need to handle the reasoning_format parameter
	// This would ideally be implemented in the go-openai library
	// For now, we'll use the standard OpenAI client but with awareness
	// that reasoning_format may need to be added to requests manually in the future

	if req.Stream {
		return handleOpenAIOpenAIStreamRequest(c, openaiClient, ctx, req, clientModel)
	}

	return handleOpenAIStandardRequest(c, openaiClient, ctx, req, clientModel)
}
