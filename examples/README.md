# Examples Directory

This directory contains example use cases for organizing and demonstrating the RAGify project's capabilities across various domains and applications.

## Purpose

The examples directory provides structured examples for:

- Knowledge bases with domain-specific content
- Documents containing realistic sample data
- Applications configured for different use cases
- Loading and managing example data
- Testing example functionality

## Example Applications

RAGify provides pre-configured examples demonstrating retrieval-augmented generation across different business and educational domains. Each example includes:

- JSON configuration files defining the application setup
- Sample documents with realistic content
- Sample questions to test the RAG system's capabilities

### Business Applications

#### 1. API Documentation Assistant
**Config:** `api_documentation.json`
**Documents:** API Reference, SDK Guides, Code Examples
**Use Case:** Helps developers understand and implement APIs and SDKs
**Benefits:** Provides instant access to documentation, reduces development time, ensures consistent API usage
**Sample Questions:**
- How do I authenticate to the /users endpoint?
- Show me a Python example for the create_widget function
- What are the API rate limits?

#### 2. Codebase Q&A Assistant
**Config:** `codebase_qa.json`
**Documents:** Source code, architecture diagrams, README files
**Use Case:** Enables natural language queries about codebase structure and functionality
**Benefits:** Accelerates onboarding, improves code understanding, supports refactoring decisions
**Sample Questions:**
- How does the authentication system work?
- What are the main components of the architecture?
- How do I add a new feature to the user management module?

#### 3. Customer Support Assistant
**Config:** `customer_support.json`
**Documents:** FAQ, troubleshooting guides, product manuals
**Use Case:** Provides instant answers to common customer inquiries
**Benefits:** Reduces support ticket volume, improves response time, ensures consistent answers
**Sample Questions:**
- How do I reset my password?
- What are the system requirements?
- How do I troubleshoot connection issues?

#### 4. HR Assistant
**Config:** `hr_assistant.json`
**Documents:** Employee handbook, company policies, benefits guide
**Use Case:** Answers employee questions about HR policies and procedures
**Benefits:** Self-service HR support, consistent policy communication, reduced administrative workload
**Sample Questions:**
- What is the maternity leave policy?
- How do I submit a vacation request?
- What benefits am I eligible for?

#### 5. Legal Analysis Assistant
**Config:** `legal_analysis.json`
**Documents:** Compliance regulations, service agreements, case studies
**Use Case:** Assists with legal document analysis and compliance questions
**Benefits:** Faster legal research, improved compliance, reduced legal consultation costs
**Sample Questions:**
- What are the GDPR requirements for data retention?
- How does this contract clause affect our liability?
- What regulatory changes affect our industry?

#### 6. Sales Enablement Assistant
**Config:** `sales_enablement.json`
**Documents:** Product specs, pricing guides, marketing brochures
**Use Case:** Provides sales teams with instant access to product information
**Benefits:** Improved sales efficiency, consistent product messaging, faster deal closure
**Sample Questions:**
- What are the key features of our premium plan?
- How does our pricing compare to competitors?
- What case studies demonstrate our ROI?

### Education & Research Applications

#### 7. Research Paper Assistant
**Config:** `research_paper_assistant.json`
**Documents:** ML Paper, History Research Paper, Biology Research Paper
**Use Case:** Helps researchers analyze and understand academic papers across disciplines
**Benefits:** Accelerates literature review, enables cross-disciplinary insights, supports academic writing
**How it works:**
1. Documents contain scholarly content with citations and methodology details
2. RAG system retrieves relevant sections based on queries
3. Provides contextual answers with source references
4. Handles complex queries about methodologies, citations, and findings
**Sample Questions:**
- Summarize the key findings from 'Attention is All You Need'
- What are the methodology differences between the machine learning and biology papers?
- How does this paper cite Dr. Smith's 2021 study?
**Edge Cases:** Handles queries requiring synthesis across multiple papers, manages technical terminology, provides accurate citations

#### 8. Interactive Study Guide
**Config:** `study_guide.json`
**Documents:** Textbook Chapter, Lecture Notes, Reading Assignment
**Use Case:** Interactive AI study guide for students learning various subjects
**Benefits:** Personalized learning support, instant clarification, comprehensive coverage of topics
**How it works:**
1. Integrates textbook content, lecture notes, and assignments
2. Provides explanations, summaries, and connections between concepts
3. Answers questions about historical events, scientific processes, and literary themes
4. Supports different learning styles with varied content types
**Sample Questions:**
- Explain the process of photosynthesis
- What are the key dates of the American Revolution?
- What are the main themes in this literature chapter?
**Edge Cases:** Manages interdisciplinary connections, handles timeline-based queries, explains complex scientific diagrams through text

## Structure

- `data/`: Contains example data files
  - `applications/`: JSON configuration files for each example application
  - `documents/`: Text files containing sample document content
- `scripts/`: Python scripts for managing examples
  - `load_examples.py`: Script to load example data into the system
  - `reset_examples.py`: Script to reset/remove example data
  - `validate_examples.py`: Script to validate example data integrity
- `tests/`: Test files for example functionality
  - `test_example_loading.py`: Tests for loading examples
  - `test_example_functionality.py`: Tests for example functionality

## Usage

To load all examples into your RAGify instance:

```bash
python examples/scripts/load_examples.py
```

To reset the examples:

```bash
python examples/scripts/reset_examples.py
```

To validate example data:

```bash
python examples/scripts/validate_examples.py
```

Each example demonstrates how RAGify can be configured for specific domains, showing the benefits of retrieval-augmented generation over traditional approaches by providing contextually relevant, source-attributed answers to complex queries.