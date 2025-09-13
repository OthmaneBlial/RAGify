# Security Policy

## Supported Versions

We actively support the following versions of RAGify with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in RAGify, please help us by reporting it responsibly. We appreciate your efforts to keep our users safe and will work with you to resolve the issue promptly.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by emailing our security team at:

**security@ragify.dev**

Alternatively, you can use GitHub's private vulnerability reporting feature:

1. Go to the [Security tab](https://github.com/OthmaneBlial/RAGify/security) in this repository
2. Click "Report a vulnerability"
3. Fill out the form with detailed information about the vulnerability

### What to Include in Your Report

To help us understand and address the vulnerability effectively, please include the following information:

- **Description**: A clear description of the vulnerability
- **Steps to Reproduce**: Detailed steps to reproduce the issue
- **Impact**: Potential impact and severity of the vulnerability
- **Affected Versions**: Which versions of RAGify are affected
- **Environment**: Your environment details (OS, Python version, etc.)
- **Proof of Concept**: If possible, include a proof of concept or exploit code
- **Contact Information**: How we can reach you for follow-up questions

### Our Response Process

We will acknowledge your report within **48 hours** and provide a more detailed response within **7 days** indicating our next steps.

We will keep you informed about our progress throughout the process of fixing the vulnerability. Once the vulnerability is fixed, we will:

1. Notify you that the fix has been completed
2. Provide a timeline for when the fix will be released
3. Credit you (if you wish) in the release notes

### Disclosure Policy

- We follow a coordinated disclosure process
- We will not publicly disclose the vulnerability until a fix is available
- We will not disclose your identity without your explicit permission
- We may share information about the vulnerability with trusted third parties (e.g., OpenRouter) if necessary for the fix

### Security Updates

Security updates will be released as soon as possible after a fix is developed and tested. We will:

- Create a new release with the security fix
- Update the changelog with details about the fix
- Notify users through our release notes and documentation

### Best Practices for Secure Usage

While we work to address any vulnerabilities, here are some security best practices for using RAGify:

#### API Keys and Secrets
- Never commit API keys or secrets to version control
- Use environment variables for sensitive configuration
- Rotate API keys regularly
- Use the minimum required permissions for API keys

#### Database Security
- Use strong passwords for database users
- Enable SSL/TLS for database connections
- Regularly backup your data
- Limit database user privileges

#### Network Security
- Run RAGify behind a reverse proxy (e.g., nginx) in production
- Enable HTTPS/TLS for all connections
- Configure proper CORS settings
- Use firewalls to restrict access

#### Application Security
- Keep dependencies updated
- Monitor application logs for suspicious activity
- Implement rate limiting
- Use secure session management

### Contact Information

For security-related questions or concerns:

- **GitHub Security Tab**: https://github.com/OthmaneBlial/RAGify/security

### Acknowledgments

We appreciate the security research community for helping keep our users safe. With your permission, we will acknowledge your contribution in our release notes.

Thank you for helping make RAGify more secure!
