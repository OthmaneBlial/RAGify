# E-Commerce API

A FastAPI-based e-commerce application with user authentication, order management, and payment processing.

## Features

- User registration and authentication with JWT tokens
- Product catalog management
- Shopping cart functionality
- Order processing and management
- Payment integration with Stripe
- Admin dashboard for order management
- RESTful API design

## Tech Stack

- **Backend**: FastAPI (Python)
- **Database**: SQLite (development) / PostgreSQL (production)
- **ORM**: SQLAlchemy
- **Authentication**: JWT tokens with bcrypt password hashing
- **Payments**: Stripe API
- **Documentation**: Auto-generated API docs with Swagger/OpenAPI

## Project Structure

```
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Application configuration
│   ├── database.py             # Database connection and session management
│   ├── models.py               # SQLAlchemy models
│   ├── schemas.py              # Pydantic schemas for request/response validation
│   ├── services/
│   │   ├── __init__.py
│   │   ├── auth_service.py     # User authentication service
│   │   └── payment_service.py  # Payment processing service
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── auth.py             # Authentication routes
│   │   ├── users.py            # User management routes
│   │   ├── products.py         # Product management routes
│   │   ├── orders.py           # Order management routes
│   │   └── payments.py         # Payment routes
│   └── utils/
│       ├── __init__.py
│       └── security.py         # Security utilities (password hashing, etc.)
├── tests/
│   ├── __init__.py
│   ├── test_auth.py
│   ├── test_orders.py
│   └── test_payments.py
├── requirements.txt
├── .env.example
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- SQLite (for development) or PostgreSQL (for production)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ecommerce-api.git
cd ecommerce-api
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Run database migrations:
```bash
alembic upgrade head
```

6. Start the development server:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Key Components

### UserAuthenticationService

The `UserAuthenticationService` class handles all user authentication-related operations:

- Password hashing and verification using bcrypt
- JWT token generation and validation
- User registration and login
- Token-based authentication middleware

### Database Connection

The application uses SQLAlchemy for database operations:

- Connection pooling for performance
- Session management with dependency injection
- Support for both SQLite (development) and PostgreSQL (production)
- Automatic table creation and migrations with Alembic

### Payment Processing

Payment processing is handled through Stripe integration:

- Secure payment intent creation
- Webhook handling for payment confirmation
- Refund processing
- Transaction logging and status tracking

## Environment Variables

```env
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///./app.db
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## Testing

Run the test suite:

```bash
pytest
```

## Deployment

### Docker

Build and run with Docker:

```bash
docker build -t ecommerce-api .
docker run -p 8000:8000 ecommerce-api
```

### Production Deployment

For production deployment:

1. Use a production WSGI server like Gunicorn
2. Set up a reverse proxy with Nginx
3. Use PostgreSQL database
4. Configure proper environment variables
5. Set up SSL/TLS certificates
6. Configure Stripe webhooks with production keys

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.