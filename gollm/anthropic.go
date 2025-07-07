// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package gollm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"k8s.io/klog/v2"
)

// Package-level env var storage (Anthropic env)
var (
	anthropicAPIKey string
	anthropicModel  string
)

// init reads and caches Anthropic environment variables:
//   - ANTHROPIC_API_KEY, ANTHROPIC_MODEL
//
// These serve as defaults; the model can be overridden by the --model flag.
// After loading env values, it registers the Anthropic provider factory.
func init() {
	// Load environment variables
	anthropicAPIKey = os.Getenv("ANTHROPIC_API_KEY")
	anthropicModel = os.Getenv("ANTHROPIC_MODEL")

	// Register "anthropic" as the provider ID
	if err := RegisterProvider("anthropic", newAnthropicClientFactory); err != nil {
		klog.Fatalf("Failed to register anthropic provider: %v", err)
	}

	// Also register with aliases
	aliases := []string{"claude"}
	for _, alias := range aliases {
		if err := RegisterProvider(alias, newAnthropicClientFactory); err != nil {
			klog.Warningf("Failed to register anthropic provider alias %q: %v", alias, err)
		}
	}
}

// AnthropicClient implements the gollm.Client interface for Anthropic models.
type AnthropicClient struct {
	client anthropic.Client
}

// Ensure AnthropicClient implements the Client interface.
var _ Client = &AnthropicClient{}

// NewAnthropicClient creates a new client for interacting with Anthropic.
// Supports custom HTTP client (e.g., for skipping SSL verification).
func NewAnthropicClient(ctx context.Context, opts ClientOptions) (*AnthropicClient, error) {
	// Get API key from loaded env var
	apiKey := anthropicAPIKey
	if apiKey == "" {
		return nil, errors.New("Anthropic API key not found. Set via ANTHROPIC_API_KEY env var")
	}

	// Set options for client creation
	options := []option.RequestOption{option.WithAPIKey(apiKey)}

	// Support custom HTTP client (e.g., skip SSL verification)
	httpClient := createCustomHTTPClient(opts.SkipVerifySSL)
	options = append(options, option.WithHTTPClient(httpClient))

	return &AnthropicClient{
		client: anthropic.NewClient(options...),
	}, nil
}

// Close cleans up any resources used by the client.
func (c *AnthropicClient) Close() error {
	// No specific cleanup needed for the Anthropic client currently.
	return nil
}

// StartChat starts a new chat session.
func (c *AnthropicClient) StartChat(systemPrompt, model string) Chat {
	// Get the model to use for this chat
	selectedModel := getAnthropicModel(model)

	klog.V(1).Infof("Starting new Anthropic chat session with model: %s", selectedModel)

	return &anthropicChatSession{
		client:       c.client,
		systemPrompt: systemPrompt,
		history:      []anthropic.MessageParam{},
		model:        selectedModel,
	}
}

// simpleAnthropicCompletionResponse is a basic implementation of CompletionResponse.
type simpleAnthropicCompletionResponse struct {
	content string
	usage   anthropic.Usage
}

// Response returns the completion content.
func (r *simpleAnthropicCompletionResponse) Response() string {
	return r.content
}

// UsageMetadata returns the usage metadata.
func (r *simpleAnthropicCompletionResponse) UsageMetadata() any {
	return r.usage
}

// GenerateCompletion sends a completion request to the Anthropic API.
func (c *AnthropicClient) GenerateCompletion(ctx context.Context, req *CompletionRequest) (CompletionResponse, error) {
	klog.Infof("Anthropic GenerateCompletion called with model: %s", req.Model)
	klog.V(1).Infof("Prompt:\n%s", req.Prompt)

	// Create the messages for the completion
	messages := []anthropic.MessageParam{
		anthropic.NewUserMessage(anthropic.TextBlockParam{
			Text: req.Prompt,
		}),
	}

	// Use the Messages API
	message, err := c.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.Model(req.Model),
		Messages:  messages,
		MaxTokens: 4096,
	})

	if err != nil {
		return nil, fmt.Errorf("failed to generate Anthropic completion: %w", err)
	}

	// Check if there are content blocks
	if len(message.Content) == 0 {
		return nil, errors.New("received an empty response from Anthropic")
	}

	// Extract text from content blocks
	var content strings.Builder
	for _, block := range message.Content {
		if textBlock, ok := block.AsTextBlock(); ok {
			content.WriteString(textBlock.Text)
		}
	}

	// Return the content
	resp := &simpleAnthropicCompletionResponse{
		content: content.String(),
		usage:   message.Usage,
	}

	return resp, nil
}

// SetResponseSchema is not implemented yet.
func (c *AnthropicClient) SetResponseSchema(schema *Schema) error {
	klog.Warning("AnthropicClient.SetResponseSchema is not implemented yet")
	return nil
}

// ListModels returns a slice of strings with model IDs.
func (c *AnthropicClient) ListModels(ctx context.Context) ([]string, error) {
	// Anthropic doesn't have a models endpoint, so we return the known models
	return []string{
		"claude-3-5-sonnet-20241022",
		"claude-3-5-haiku-20241022",
		"claude-3-opus-20240229",
		"claude-3-sonnet-20240229",
		"claude-3-haiku-20240307",
	}, nil
}

// Chat Session Implementation

type anthropicChatSession struct {
	client              anthropic.Client
	systemPrompt        string
	history             []anthropic.MessageParam
	model               string
	functionDefinitions []*FunctionDefinition
	tools               []anthropic.ToolParam
}

// Ensure anthropicChatSession implements the Chat interface.
var _ Chat = (*anthropicChatSession)(nil)

// SetFunctionDefinitions stores the function definitions and converts them to Anthropic format.
func (cs *anthropicChatSession) SetFunctionDefinitions(defs []*FunctionDefinition) error {
	cs.functionDefinitions = defs
	cs.tools = nil // Clear previous tools

	if len(defs) > 0 {
		cs.tools = make([]anthropic.ToolParam, len(defs))
		for i, gollmDef := range defs {
			klog.Infof("Processing function definition: %s", gollmDef.Name)

			// Convert function parameters to Anthropic format
			inputSchema, err := cs.convertFunctionParameters(gollmDef)
			if err != nil {
				return fmt.Errorf("failed to process parameters for function %s: %w", gollmDef.Name, err)
			}

			cs.tools[i] = anthropic.ToolParam{
				Name:        gollmDef.Name,
				Description: anthropic.String(gollmDef.Description),
				InputSchema: inputSchema,
			}
		}
	}

	klog.V(1).Infof("Set %d function definitions for Anthropic chat session", len(cs.functionDefinitions))
	return nil
}

// Send sends the user message(s), appends to history, and gets the LLM response.
func (cs *anthropicChatSession) Send(ctx context.Context, contents ...any) (ChatResponse, error) {
	klog.V(1).InfoS("anthropicChatSession.Send called", "model", cs.model, "history_len", len(cs.history))

	// Process and append messages to history
	if err := cs.addContentsToHistory(contents); err != nil {
		return nil, err
	}

	// Prepare and send API request
	messageParams := anthropic.MessageNewParams{
		Model:     anthropic.Model(cs.model),
		Messages:  cs.history,
		MaxTokens: 4096,
	}

	// Add system prompt if provided
	if cs.systemPrompt != "" {
		messageParams.System = []anthropic.TextBlockParam{
			{Text: cs.systemPrompt},
		}
	}

	// Add tools if available
	if len(cs.tools) > 0 {
		messageParams.Tools = cs.tools
	}

	// Call the Anthropic API
	klog.V(1).InfoS("Sending request to Anthropic Messages API", "model", cs.model, "messages", len(messageParams.Messages), "tools", len(messageParams.Tools))
	message, err := cs.client.Messages.New(ctx, messageParams)
	if err != nil {
		klog.Errorf("Anthropic Messages API error: %v", err)
		return nil, fmt.Errorf("Anthropic message completion failed: %w", err)
	}
	klog.V(1).InfoS("Received response from Anthropic Messages API", "id", message.ID, "content_blocks", len(message.Content))

	// Process the response
	if len(message.Content) == 0 {
		klog.Warning("Received response with no content from Anthropic")
		return nil, errors.New("received empty response from Anthropic (no content)")
	}

	// Add assistant's response to history - convert to param format
	var contentBlocks []anthropic.ContentBlockParamUnion
	for _, block := range message.Content {
		contentBlocks = append(contentBlocks, block.ToParam())
	}
	
	assistantMessage := anthropic.NewAssistantMessage(contentBlocks...)
	cs.history = append(cs.history, assistantMessage)
	klog.V(2).InfoS("Added assistant message to history", "content_blocks", len(message.Content))

	// Wrap the response
	resp := &anthropicChatResponse{
		anthropicMessage: message,
	}

	return resp, nil
}

// SendStreaming sends the user message(s) and returns an iterator for the LLM response stream.
func (cs *anthropicChatSession) SendStreaming(ctx context.Context, contents ...any) (ChatResponseIterator, error) {
	klog.V(1).InfoS("Starting Anthropic streaming request", "model", cs.model)

	// Process and append messages to history
	if err := cs.addContentsToHistory(contents); err != nil {
		return nil, err
	}

	// Prepare and send API request
	messageParams := anthropic.MessageNewParams{
		Model:     anthropic.Model(cs.model),
		Messages:  cs.history,
		MaxTokens: 4096,
	}

	// Add system prompt if provided
	if cs.systemPrompt != "" {
		messageParams.System = []anthropic.TextBlockParam{
			{Text: cs.systemPrompt},
		}
	}

	// Add tools if available
	if len(cs.tools) > 0 {
		messageParams.Tools = cs.tools
	}

	// Start the Anthropic streaming request
	klog.V(1).InfoS("Sending streaming request to Anthropic API",
		"model", cs.model,
		"messageCount", len(messageParams.Messages),
		"toolCount", len(messageParams.Tools))

	stream := cs.client.Messages.NewStreaming(ctx, messageParams)

	// Create and return the stream iterator
	return func(yield func(ChatResponse, error) bool) {
		defer stream.Close()

		var contentBuilder strings.Builder

		// Process stream events
		for stream.Next() {
			event := stream.Current()
			
			// Handle different event types based on the event
			if textDelta := event.Delta.AsTextDelta(); textDelta != nil {
				contentBuilder.WriteString(textDelta.Text)
				
				// Create streaming response
				streamResponse := &anthropicChatStreamResponse{
					content: textDelta.Text,
				}

				if !yield(streamResponse, nil) {
					return
				}
			}
		}

		// Check for errors after streaming completes
		if err := stream.Err(); err != nil {
			klog.Errorf("Error in Anthropic streaming: %v", err)
			yield(nil, fmt.Errorf("Anthropic streaming error: %w", err))
			return
		}

		// Update conversation history with the complete message
		if contentBuilder.Len() > 0 {
			assistantMessage := anthropic.NewAssistantMessage(anthropic.TextBlockParam{
				Text: contentBuilder.String(),
			})
			cs.history = append(cs.history, assistantMessage)
			klog.V(2).InfoS("Added complete assistant message to history")
		}
	}, nil
}

// IsRetryableError determines if an error from the Anthropic API should be retried.
func (cs *anthropicChatSession) IsRetryableError(err error) bool {
	if err == nil {
		return false
	}
	return DefaultIsRetryableError(err)
}

// Helper structs for ChatResponse interface

type anthropicChatResponse struct {
	anthropicMessage *anthropic.Message
}

var _ ChatResponse = (*anthropicChatResponse)(nil)

func (r *anthropicChatResponse) UsageMetadata() any {
	if r.anthropicMessage != nil {
		return r.anthropicMessage.Usage
	}
	return nil
}

func (r *anthropicChatResponse) Candidates() []Candidate {
	if r.anthropicMessage == nil {
		return nil
	}
	// Anthropic returns a single response, so we create one candidate
	return []Candidate{&anthropicCandidate{anthropicMessage: r.anthropicMessage}}
}

type anthropicCandidate struct {
	anthropicMessage *anthropic.Message
}

var _ Candidate = (*anthropicCandidate)(nil)

func (c *anthropicCandidate) Parts() []Part {
	if c.anthropicMessage == nil {
		return nil
	}

	var parts []Part
	var textContent strings.Builder
	var toolUses []anthropic.ContentBlockUnion

	// Process content blocks
	for _, block := range c.anthropicMessage.Content {
		if textBlock, ok := block.AsTextBlock(); ok {
			textContent.WriteString(textBlock.Text)
		} else if toolUse, ok := block.AsToolUseBlock(); ok {
			toolUses = append(toolUses, block)
		}
	}

	// Add text part if there's content
	if textContent.Len() > 0 {
		parts = append(parts, &anthropicPart{content: textContent.String()})
	}

	// Add tool use part if there are tool uses
	if len(toolUses) > 0 {
		parts = append(parts, &anthropicPart{toolUses: toolUses})
	}

	return parts
}

// String provides a simple string representation for logging/debugging.
func (c *anthropicCandidate) String() string {
	if c.anthropicMessage == nil {
		return "<nil candidate>"
	}
	
	var contentSummary strings.Builder
	toolUseCount := 0
	
	for _, block := range c.anthropicMessage.Content {
		if textBlock, ok := block.AsTextBlock(); ok {
			if contentSummary.Len() > 0 {
				contentSummary.WriteString("; ")
			}
			contentSummary.WriteString(textBlock.Text)
		} else if _, ok := block.AsToolUseBlock(); ok {
			toolUseCount++
		}
	}
	
	content := "<no content>"
	if contentSummary.Len() > 0 {
		content = contentSummary.String()
	}
	
	stopReason := string(c.anthropicMessage.StopReason)
	return fmt.Sprintf("Candidate(StopReason: %s, ToolUses: %d, Content: %q)", stopReason, toolUseCount, content)
}

type anthropicPart struct {
	content  string
	toolUses []anthropic.ContentBlockUnion
}

var _ Part = (*anthropicPart)(nil)

func (p *anthropicPart) AsText() (string, bool) {
	return p.content, p.content != ""
}

func (p *anthropicPart) AsFunctionCalls() ([]FunctionCall, bool) {
	return convertAnthropicToolUsesToFunctionCalls(p.toolUses)
}

// Streaming response implementation
type anthropicChatStreamResponse struct {
	content string
}

var _ ChatResponse = (*anthropicChatStreamResponse)(nil)

func (r *anthropicChatStreamResponse) UsageMetadata() any {
	return nil // Usage is not available during streaming
}

func (r *anthropicChatStreamResponse) Candidates() []Candidate {
	return []Candidate{&anthropicStreamCandidate{content: r.content}}
}

type anthropicStreamCandidate struct {
	content string
}

var _ Candidate = (*anthropicStreamCandidate)(nil)

func (c *anthropicStreamCandidate) Parts() []Part {
	if c.content == "" {
		return nil
	}
	return []Part{&anthropicStreamPart{content: c.content}}
}

func (c *anthropicStreamCandidate) String() string {
	return fmt.Sprintf("StreamingCandidate(Content: %q)", c.content)
}

type anthropicStreamPart struct {
	content string
}

var _ Part = (*anthropicStreamPart)(nil)

func (p *anthropicStreamPart) AsText() (string, bool) {
	return p.content, p.content != ""
}

func (p *anthropicStreamPart) AsFunctionCalls() ([]FunctionCall, bool) {
	return nil, false // Tool calls are not processed during streaming deltas
}

// Helper functions

// newAnthropicClientFactory is the factory function for creating Anthropic clients.
func newAnthropicClientFactory(ctx context.Context, opts ClientOptions) (Client, error) {
	return NewAnthropicClient(ctx, opts)
}

// addContentsToHistory processes and appends user messages to chat history
func (cs *anthropicChatSession) addContentsToHistory(contents []any) error {
	for _, content := range contents {
		switch c := content.(type) {
		case string:
			klog.V(2).Infof("Adding user message to history: %s", c)
			cs.history = append(cs.history, anthropic.NewUserMessage(anthropic.TextBlockParam{
				Text: c,
			}))
		case FunctionCallResult:
			klog.V(2).Infof("Adding tool call result to history: Name=%s, ID=%s", c.Name, c.ID)
			// Marshal the result map into a JSON string for the message content
			resultJSON, err := json.Marshal(c.Result)
			if err != nil {
				klog.Errorf("Failed to marshal function call result: %v", err)
				return fmt.Errorf("failed to marshal function call result %q: %w", c.Name, err)
			}
			
			cs.history = append(cs.history, anthropic.NewUserMessage(anthropic.ToolResultBlockParam{
				ToolUseID: c.ID,
				Content:   string(resultJSON),
			}))
		default:
			klog.Warningf("Unhandled content type: %T", content)
			return fmt.Errorf("unhandled content type: %T", content)
		}
	}
	return nil
}

// convertAnthropicToolUsesToFunctionCalls converts Anthropic tool uses to gollm function calls
func convertAnthropicToolUsesToFunctionCalls(contentBlocks []anthropic.ContentBlockUnion) ([]FunctionCall, bool) {
	var calls []FunctionCall
	
	for _, block := range contentBlocks {
		if toolUseBlock, ok := block.AsToolUseBlock(); ok {
			calls = append(calls, FunctionCall{
				ID:        toolUseBlock.ID,
				Name:      toolUseBlock.Name,
				Arguments: toolUseBlock.Input,
			})
		}
	}
	
	return calls, len(calls) > 0
}

// convertFunctionParameters handles the conversion of gollm parameters to Anthropic format
func (cs *anthropicChatSession) convertFunctionParameters(gollmDef *FunctionDefinition) (anthropic.ToolInputSchemaParam, error) {
	if gollmDef.Parameters == nil {
		return anthropic.ToolInputSchemaParam{
			"type":       "object",
			"properties": map[string]interface{}{},
		}, nil
	}

	// Convert the schema to JSON and back to interface{} for Anthropic format
	schemaBytes, err := json.Marshal(gollmDef.Parameters)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal schema: %w", err)
	}

	var inputSchema anthropic.ToolInputSchemaParam
	if err := json.Unmarshal(schemaBytes, &inputSchema); err != nil {
		return nil, fmt.Errorf("failed to unmarshal schema: %w", err)
	}

	return inputSchema, nil
}

// getAnthropicModel returns the appropriate model based on configuration and explicitly provided model name
func getAnthropicModel(model string) string {
	// If explicit model is provided, use it
	if model != "" {
		klog.V(2).Infof("Using explicitly provided model: %s", model)
		return model
	}

	// Check configuration
	configModel := anthropicModel
	if configModel != "" {
		klog.V(1).Infof("Using model from config: %s", configModel)
		return configModel
	}

	// Default model as fallback
	klog.V(2).Info("No model specified, defaulting to claude-3-5-sonnet-20241022")
	return "claude-3-5-sonnet-20241022"
}