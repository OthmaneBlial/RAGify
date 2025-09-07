# API Documentation

This document provides comprehensive documentation for the RAGify REST API. The API is built with FastAPI and provides endpoints for managing knowledge bases, documents, applications, and chat functionality.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, RAGify does not implement authentication. In production deployments, consider adding:

- JWT token authentication
- API key authentication
- OAuth 2.0 integration

## Response Format

All API responses follow this structure:

```json
{
  "data": {...},  // Response data
  "message": "Success",  // Optional message
  "errors": null  // Error details if any
}
```

## Error Handling

### HTTP Status Codes

- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `404` - Not Found
- `422` - Validation Error
- `500` - Internal Server Error

### Error Response Format

```json
{
  "detail": "Error message",
  "errors": [
    {
      "field": "field_name",
      "message": "Validation error message"
    }
  ]
}
```

## Knowledge Base Endpoints

### Create Knowledge Base

**POST** `/knowledge-bases/`

Creates a new knowledge base.

**Request Body:**

```json
{
  "name": "Company Documentation",
  "description": "Internal company documents and procedures"
}
```

**Response (201):**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Company Documentation",
  "description": "Internal company documents and procedures",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

### List Knowledge Bases

**GET** `/knowledge-bases/`

Retrieves all knowledge bases.

**Response (200):**

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Company Documentation",
    "description": "Internal company documents",
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:00Z"
  }
]
```

### Get Knowledge Base

**GET** `/knowledge-bases/{kb_id}`

Retrieves a specific knowledge base.

**Parameters:**

- `kb_id` (UUID) - Knowledge base ID

**Response (200):**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Company Documentation",
  "description": "Internal company documents",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

### Update Knowledge Base

**PUT** `/knowledge-bases/{kb_id}`

Updates a knowledge base.

**Parameters:**

- `kb_id` (UUID) - Knowledge base ID

**Request Body:**

```json
{
  "name": "Updated Documentation",
  "description": "Updated description"
}
```

### Delete Knowledge Base

**DELETE** `/knowledge-bases/{kb_id}`

Deletes a knowledge base.

**Parameters:**

- `kb_id` (UUID) - Knowledge base ID

**Response (200):**

```json
{
  "message": "Knowledge base deleted"
}
```

## Document Endpoints

### Upload Document

**POST** `/knowledge-bases/{kb_id}/documents/`

Uploads a document to a knowledge base.

**Parameters:**

- `kb_id` (UUID) - Knowledge base ID

**Request:**

- Content-Type: `multipart/form-data`
- File: Document file (PDF, DOCX, TXT)

**Response (201):**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440001",
  "title": "user_manual.pdf",
  "content": "Extracted text content...",
  "knowledge_base_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2024-01-15T10:35:00Z",
  "updated_at": "2024-01-15T10:35:00Z",
  "processing_status": "completed"
}
```

### List Documents

**GET** `/knowledge-bases/{kb_id}/documents/`

Lists all documents in a knowledge base.

**Parameters:**

- `kb_id` (UUID) - Knowledge base ID

**Response (200):**

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440001",
    "title": "user_manual.pdf",
    "knowledge_base_id": "550e8400-e29b-41d4-a716-446655440000",
    "created_at": "2024-01-15T10:35:00Z",
    "processing_status": "completed"
  }
]
```

### Get Document Status

**GET** `/documents/{doc_id}/status`

Gets the processing status of a document.

**Parameters:**

- `doc_id` (UUID) - Document ID

**Response (200):**

```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440001",
  "processing_status": "completed",
  "title": "user_manual.pdf",
  "created_at": "2024-01-15T10:35:00Z",
  "updated_at": "2024-01-15T10:35:00Z"
}
```

## Search Endpoints

### Search Knowledge Bases

**POST** `/search/`

Performs semantic search across knowledge bases.

**Request Body:**

```json
{
  "query": "How do I reset my password?",
  "knowledge_base_id": "550e8400-e29b-41d4-a716-446655440000", // Optional
  "limit": 10,
  "threshold": 0.7
}
```

**Response (200):**

```json
{
  "results": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440002",
      "content": "To reset your password, go to settings...",
      "similarity_score": 0.85,
      "document_id": "550e8400-e29b-41d4-a716-446655440001",
      "knowledge_base_id": "550e8400-e29b-41d4-a716-446655440000"
    }
  ],
  "query": "How do I reset my password?",
  "total_results": 1,
  "knowledge_base_id": "550e8400-e29b-41d4-a716-446655440000",
  "threshold": 0.7
}
```

## Application Endpoints

### Create Application

**POST** `/applications/`

Creates a new chat application.

**Request Body:**

```json
{
  "name": "Customer Support Bot",
  "description": "AI assistant for customer inquiries",
  "model_config": {
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 1000
  },
  "knowledge_base_ids": [
    "550e8400-e29b-41d4-a716-446655440000"
  ]
}
```

**Response (201):**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440003",
  "name": "Customer Support Bot",
  "description": "AI assistant for customer inquiries",
  "model_config": {
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 1000
  },
  "knowledge_base_ids": [
    "550e8400-e29b-41d4-a716-446655440000"
  ],
  "created_at": "2024-01-15T11:00:00Z",
  "updated_at": "2024-01-15T11:00:00Z"
}
```

### List Applications

**GET** `/applications/`

Lists all applications.

**Response (200):**

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440003",
    "name": "Customer Support Bot",
    "description": "AI assistant for customer inquiries",
    "model_config": {...},
    "knowledge_base_ids": [...],
    "created_at": "2024-01-15T11:00:00Z",
    "updated_at": "2024-01-15T11:00:00Z"
  }
]
```

### Get Application

**GET** `/applications/{application_id}`

Retrieves a specific application.

**Parameters:**

- `application_id` (UUID) - Application ID

### Update Application

**PUT** `/applications/{application_id}`

Updates an application.

**Parameters:**

- `application_id` (UUID) - Application ID

### Delete Application

**DELETE** `/applications/{application_id}`

Deletes an application.

**Parameters:**

- `application_id` (UUID) - Application ID

### Chat with Application

**POST** `/applications/{application_id}/chat`

Sends a chat message to an application.

**Parameters:**

- `application_id` (UUID) - Application ID

**Request Body:**

```json
{
  "message": "How do I reset my password?",
  "search_type": "hybrid",
  "max_context_length": 4000,
  "temperature": 0.7,
  "stream": false
}
```

**Response (200):**

```json
{
  "response": "To reset your password, go to the settings page and click 'Reset Password'...",
  "context_count": 3,
  "confidence_score": 0.85,
  "metadata": {
    "model": "gpt-3.5-turbo",
    "tokens_used": 150,
    "processing_time": 1.2
  }
}
```

### Get Chat History

**GET** `/applications/{application_id}/chat/history`

Retrieves chat history for an application.

**Parameters:**

- `application_id` (UUID) - Application ID
- `limit` (int, optional) - Maximum messages to return (default: 50)

**Response (200):**

```json
{
  "messages": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440004",
      "user_message": "How do I reset my password?",
      "bot_message": "To reset your password...",
      "created_at": "2024-01-15T11:05:00Z"
    }
  ]
}
```

## Chat Endpoints

### Send Chat Message

**POST** `/chat/message`

Sends a chat message (non-streaming).

**Request Body:**

```json
{
  "message": "Hello, how can I help you?",
  "application_id": "550e8400-e29b-41d4-a716-446655440003",
  "search_type": "hybrid",
  "max_context_length": 4000,
  "temperature": 0.7,
  "stream": false
}
```

**Response (200):**

```json
{
  "message_id": "550e8400-e29b-41d4-a716-446655440005",
  "response": "Hello! I'm here to help you with any questions...",
  "context_count": 2,
  "confidence_score": 0.75,
  "metadata": {...}
}
```

### Send Streaming Chat Message

**POST** `/chat/message/stream`

Sends a chat message with streaming response.

**Request Body:** Same as non-streaming endpoint

**Response:** Server-sent events stream

### Get Conversation History

**GET** `/chat/history/{application_id}`

Gets conversation history for an application.

**Parameters:**

- `application_id` (UUID) - Application ID
- `limit` (int, optional) - Maximum messages (default: 50)
- `before_message_id` (UUID, optional) - Get messages before this ID

### Clear Conversation History

**DELETE** `/chat/history/{application_id}`

Clears conversation history for an application.

**Parameters:**

- `application_id` (UUID) - Application ID

### Get Chat Statistics

**GET** `/chat/applications/{application_id}/stats`

Gets chat statistics for an application.

**Parameters:**

- `application_id` (UUID) - Application ID

**Response (200):**

```json
{
  "application_id": "550e8400-e29b-41d4-a716-446655440003",
  "total_messages": 150,
  "user_messages": 75,
  "bot_messages": 75,
  "conversations": 75
}
```

## WebSocket Endpoints

### Real-time Chat

**WebSocket** `/chat/ws/{application_id}`

Establishes a WebSocket connection for real-time chat.

**Parameters:**

- `application_id` (UUID) - Application ID

**Message Format:**

```json
{
  "type": "message",
  "message": "Hello!",
  "search_type": "hybrid"
}
```

## Rate Limiting

The API implements rate limiting to prevent abuse:

- Root endpoint: 100 requests per minute
- Health check: 60 requests per minute
- Other endpoints: Configurable via environment variables

## Pagination

For endpoints that return lists, use query parameters:

- `page` (int) - Page number (default: 1)
- `per_page` (int) - Items per page (default: 20, max: 100)

Example:

```
GET /knowledge-bases?page=2&per_page=10
```

## File Upload Limits

- Maximum file size: 100MB (configurable)
- Supported formats: PDF, DOCX, TXT
- Multiple files: Not supported (upload one at a time)

## SDK Examples

### Python

```python
import requests

# Create knowledge base
kb = requests.post("http://localhost:8000/api/v1/knowledge-bases/",
    json={"name": "My KB", "description": "Test knowledge base"}
).json()

# Upload document
with open("document.pdf", "rb") as f:
    requests.post(f"http://localhost:8000/api/v1/knowledge-bases/{kb['id']}/documents/",
        files={"file": ("document.pdf", f, "application/pdf")})

# Search
results = requests.post("http://localhost:8000/api/v1/search/",
    json={"query": "search term"}).json()
```

### JavaScript

```javascript
// Using fetch API
const response = await fetch(
  "http://localhost:8000/api/v1/knowledge-bases/",
  {
    method: "POST",
    headers: {
      "Content-Type":
        "application/json",
    },
    body: JSON.stringify({
      name: "My Knowledge Base",
      description: "Description",
    }),
  }
)

const kb = await response.json()
```

## API Versioning

The API uses URL-based versioning:

- Current version: v1
- Base path: `/api/v1/`
- Future versions will be available at `/api/v2/`, etc.

## Monitoring and Health Checks

### Health Check

**GET** `/health`

Returns system health status.

**Response (200):**

```json
{
  "status": "ok",
  "timestamp": 1642156800.123,
  "database": {
    "connections": 5,
    "status": "healthy"
  },
  "cache": {
    "status": "healthy",
    "hits": 150,
    "misses": 25
  },
  "task_queue": {
    "active_tasks": 2,
    "queue_size": 10
  }
}
```

## Error Codes

| Code | Description |
| --- | --- |
| `VALIDATION_ERROR` | Invalid request data |
| `NOT_FOUND` | Resource not found |
| `PERMISSION_DENIED` | Insufficient permissions |
| `RATE_LIMIT_EXCEEDED` | Too many requests |
| `INTERNAL_ERROR` | Server error |
| `SERVICE_UNAVAILABLE` | Service temporarily unavailable |

## Best Practices

1. **Use appropriate content types** for requests
2. **Handle rate limits** by implementing exponential backoff
3. **Validate data** on the client side before sending
4. **Use streaming** for large responses
5. **Implement proper error handling** for all API calls
6. **Cache responses** when appropriate
7. **Use pagination** for large result sets

## Support

For API support and questions:

- Check the interactive API documentation at `/docs`
- Review the application logs for detailed error information
- Open an issue on the project repository
