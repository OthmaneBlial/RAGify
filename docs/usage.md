# Usage Guide

This guide provides step-by-step instructions for using RAGify to create knowledge bases, upload documents, build applications, and interact with the chat interface.

## Getting Started

After [setting up](setup.md) RAGify, access the web interface at `http://localhost:5173`.

The interface consists of three main sections:

- **Chat**: Interact with your AI applications
- **Knowledge Bases**: Manage your document collections
- **Applications**: Create and configure chat applications

## Creating Knowledge Bases

Knowledge bases are containers for your documents and serve as the foundation for your AI applications.

### Step 1: Access Knowledge Bases

1. Click on "📚 Knowledge Bases" in the sidebar
2. You'll see a list of existing knowledge bases (empty initially)

### Step 2: Create a New Knowledge Base

1. Click the "Create Knowledge Base" button
2. Fill in the details:
   - **Name**: A descriptive name (e.g., "Company Policies")
   - **Description**: Optional details about the knowledge base
3. Click "Create"

### Step 3: Upload Documents

1. Select your newly created knowledge base
2. Click "Upload Documents"
3. Choose files to upload:
   - **Supported formats**: PDF, DOCX, TXT
   - **Maximum size**: 100MB per file
   - **Multiple files**: Upload one at a time

### Step 4: Monitor Processing

After uploading, documents go through processing:

- **Status indicators**: Pending → Processing → Completed
- **Processing includes**:
  - Text extraction from files
  - Document chunking
  - Vector embedding generation
  - Indexing for search

## Setting Up Applications

Applications are chat interfaces powered by your knowledge bases.

### Step 1: Create an Application

1. Navigate to "🚀 Applications" in the sidebar
2. Click "Create Application"
3. Configure the application:
   - **Name**: Application display name
   - **Description**: Purpose of the application
   - **Model Selection**: Choose from 100+ models via OpenRouter
   - **Temperature**: Response creativity (0.0-1.0)
   - **Max tokens**: Response length limit

### Step 2: Associate Knowledge Bases

1. In the application settings, find "Knowledge Bases"
2. Select one or more knowledge bases to associate
3. Click "Save Changes"

### Step 3: Configure Chat Settings

Fine-tune your application's behavior:

- **Search Type**: Semantic, keyword, or hybrid search
- **Context Length**: Maximum context to include (1000-8000 tokens)
- **Temperature**: Response creativity (0.0-1.0)
- **System Prompt**: Custom instructions for the AI

## Using the Chat Interface

### Basic Chat

1. Go to "💬 Chat" in the sidebar
2. Select an application from the dropdown
3. Type your message in the input field
4. Press Enter or click Send

### Streaming Responses

RAGify supports real-time streaming:

- Responses appear word-by-word as generated
- Better user experience for longer responses
- Can be interrupted if needed

### Conversation History

- Previous conversations are automatically saved
- Access history through the sidebar
- Clear history if needed
- Export conversations (future feature)

## Advanced Features

### Semantic Search

RAGify uses advanced semantic search:

- **Vector embeddings** for meaning-based search
- **Hybrid search** combining semantic and keyword matching
- **Relevance scoring** to rank results
- **Context-aware responses** using retrieved information

### Document Processing

Supported document types and processing:

#### PDF Documents

- Text extraction from all pages
- Table recognition and formatting
- Image OCR (if images contain text)
- Maintains document structure

#### Word Documents (DOCX)

- Preserves formatting and structure
- Extracts text from headers, footers, and body
- Handles complex layouts

#### Text Files

- Direct text processing
- Supports various encodings
- Handles large files efficiently

### Chat Customization

#### System Prompts

Create custom system prompts to:

- Define the AI's personality
- Set response guidelines
- Include specific instructions
- Control output format

#### Model Selection

Choose from 100+ AI models available through OpenRouter:

- **OpenAI Models**: GPT-4, GPT-4o, GPT-3.5 Turbo, etc.
- **Anthropic Models**: Claude-3, Claude-2, etc.
- **Google Models**: Gemini, etc.
- **Meta Models**: Llama models
- **And many more**: From various providers

All models are accessed through OpenRouter's unified API, making it easy to switch between providers without code changes.

## Best Practices

### Knowledge Base Organization

1. **Logical Grouping**: Group related documents together
2. **Naming Conventions**: Use clear, descriptive names
3. **Size Considerations**: Balance between comprehensive and focused
4. **Regular Updates**: Keep knowledge bases current

### Document Preparation

1. **Quality Content**: Ensure documents are well-written and accurate
2. **Format Optimization**:
   - Use clear headings and structure
   - Include relevant metadata
   - Remove unnecessary formatting
3. **File Size**: Compress large files when possible
4. **Text Extraction**: Test document processing before bulk upload

### Application Configuration

1. **Purpose Definition**: Clearly define the application's role
2. **Knowledge Base Selection**: Choose relevant knowledge bases only
3. **Model Tuning**:
   - Lower temperature for factual responses
   - Higher temperature for creative tasks
   - Adjust context length based on needs

### Chat Interaction

1. **Clear Questions**: Ask specific, well-formed questions
2. **Context Provision**: Provide necessary context in your messages
3. **Follow-up Questions**: Build on previous interactions
4. **Feedback Loop**: Note what works and what doesn't

## Troubleshooting Common Issues

### Chat Responses

#### Irrelevant Answers

- **Cause**: Poor document quality or irrelevant knowledge base
- **Solution**: Review and improve source documents

#### Incomplete Responses

- **Cause**: Context length too short or model limitations
- **Solution**: Increase max context or choose different model

#### Slow Responses

- **Cause**: Large knowledge base or complex queries
- **Solution**: Optimize knowledge base size or simplify queries

### Document Processing

#### Upload Failures

- **Cause**: Unsupported format or file too large
- **Solution**: Check file format and size limits

#### Poor Text Extraction

- **Cause**: Complex formatting or scanned documents
- **Solution**: Convert to text-friendly formats or use OCR

### Application Issues

#### No Response

- **Cause**: Model API key missing or invalid
- **Solution**: Verify API keys in configuration

#### Connection Errors

- **Cause**: Backend service unavailable
- **Solution**: Check service status and logs

## Performance Optimization

### Knowledge Base Optimization

1. **Chunk Size**: Balance between context and precision
2. **Embedding Model**: Choose appropriate model for your domain
3. **Index Updates**: Regular re-indexing for updated content
4. **Caching**: Enable Redis for improved performance

### Application Tuning

1. **Model Selection**: Choose from OpenRouter's extensive model library
   - Use GPT-4 models for complex reasoning
   - Use Claude models for detailed analysis
   - Use specialized models for specific tasks
2. **Parameter Optimization**:
   - Temperature based on creativity needs
   - Max tokens based on response requirements
3. **Knowledge Base Size**: Limit to relevant documents only

### System Resources

1. **Memory Management**: Monitor RAM usage with large knowledge bases
2. **Storage**: Plan for vector database growth
3. **Network**: Optimize for API call efficiency
4. **Caching**: Use Redis for session and response caching

## Integration Examples

### API Integration

```python
import requests

# Chat with an application
response = requests.post(
    "http://localhost:8000/api/v1/applications/{app_id}/chat",
    json={"message": "How do I reset my password?"}
)

print(response.json()["response"])
```

### Webhook Integration

```javascript
// Receive chat responses via webhook
app.post(
  "/webhook/chat",
  (req, res) => {
    const {
      message,
      response,
      application_id,
    } = req.body
    // Process the chat interaction
    console.log(
      `Chat in ${application_id}: ${message} -> ${response}`
    )
  }
)
```

### Custom Frontend

```html
<!-- Embed chat widget -->
<div id="ragify-chat"></div>
<script src="https://your-domain.com/chat-widget.js"></script>
<script>
  RagifyChat.init({
    applicationId: "your-app-id",
    apiUrl:
      "http://localhost:8000/api/v1",
  })
</script>
```

## Monitoring and Analytics

### Chat Analytics

- **Message Volume**: Track conversation frequency
- **Response Quality**: Monitor user satisfaction
- **Popular Topics**: Identify common queries
- **Performance Metrics**: Response times and success rates

### Usage Statistics

- **Active Users**: Track user engagement
- **Knowledge Base Usage**: Most accessed documents
- **Application Performance**: Success rates by application
- **System Health**: Monitor resource usage

## Security Considerations

### Data Privacy

1. **Document Security**: Ensure sensitive documents are properly handled
2. **Access Control**: Implement user authentication for production
3. **Data Encryption**: Use encrypted connections and storage
4. **Audit Logging**: Track all access and modifications

### API Security

1. **Rate Limiting**: Prevent abuse with appropriate limits
2. **Input Validation**: Sanitize all user inputs
3. **Error Handling**: Don't expose sensitive information in errors
4. **CORS Configuration**: Restrict origins in production

## Support and Resources

### Getting Help

1. **Documentation**: Check this guide and API documentation
2. **Logs**: Review application logs for error details
3. **Community**: Join discussions and ask questions
4. **Issues**: Report bugs on the project repository

### Additional Resources

- [API Documentation](api.md)
- [Setup Guide](setup.md)
- [Development Guide](development.md)
- [Deployment Guide](deployment.md)

---

**Next Steps**: Explore the [API documentation](api.md) for programmatic access or the [development guide](development.md) for extending RAGify.
