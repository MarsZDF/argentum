# Security Policy

## Overview

Argentum implements comprehensive security practices to ensure safe operation in production environments. This document outlines the security features, best practices, and vulnerability reporting procedures.

## 🛡️ Security Features

### 1. Input Validation & Sanitization

All user inputs are validated and sanitized to prevent:
- **Code Injection**: No `eval()`, `exec()`, or dynamic code execution
- **XSS Attacks**: HTML/JavaScript sanitization in all string inputs  
- **Path Traversal**: File path validation prevents `../` attacks
- **Size Limits**: Resource exhaustion protection with configurable limits

```python
from argentum.security import configure_security

# Configure security limits for production
configure_security(
    max_state_size_mb=5,      # Limit state objects to 5MB
    max_context_items=5000,   # Limit context decay items
    max_plan_steps=1000,      # Limit plan complexity
    enable_all_protections=True
)
```

### 2. Secrets Detection

Automatic scanning for exposed credentials and secrets:

```python
from argentum import PlanLinter

# Built-in secrets patterns
secrets = ["api_key", "sk-", "password", "token", "secret"]
linter = PlanLinter()
result = linter.lint(plan, tool_specs, secrets=secrets)

# Check for security issues
security_issues = [i for i in result.issues if i.code == "E004"]
if security_issues:
    print("🚨 Credentials detected in plan!")
```

### 3. Resource Limits

Protection against DoS attacks through resource exhaustion:

| Resource | Default Limit | Configurable |
|----------|---------------|--------------|
| State Size | 10 MB | ✅ |
| Context Items | 10,000 | ✅ |
| Plan Steps | 1,000 | ✅ |
| Artifact Count | 100 | ✅ |
| Plan Parameters | 50 per step | ✅ |

### 4. Secure Logging

Prevents log injection and sensitive data exposure:

```python
from argentum.logging import setup_logging

# Secure logging configuration
logger = setup_logging(
    format_type="structured",    # Structured output prevents injection
    sanitize_log_data=True      # Automatic data sanitization
)

# Safe logging with context
logger.info("Processing plan", extra={
    "plan_id": plan_id,         # Safe: IDs only
    "agent_id": agent_id        # Safe: No sensitive data
})
```

### 5. Safe Serialization

Only JSON serialization - no pickle or unsafe formats:

```python
# ✅ Safe - JSON only
handoff_json = protocol.to_json(handoff)
restored = protocol.from_json(handoff_json)

# ❌ Avoided - no pickle, eval, exec
# pickle.loads(data)  # NOT used in argentum
```

### 6. Webhook Security

Comprehensive security for external webhook integrations:

- **HTTPS Enforcement**: Only HTTPS webhook URLs are accepted
- **Private Network Protection**: Blocks localhost and private IP ranges
- **Message Template Sanitization**: Prevents format string injection attacks
- **Request Timeout Controls**: 10-second timeout to prevent DoS
- **Domain Validation**: Warnings for unknown webhook services

```python
from argentum import CostAlerts

alerts = CostAlerts()

# ✅ Safe - validates URL and sanitizes message
alerts.add_webhook(
    "https://hooks.slack.com/services/...",  # HTTPS required
    threshold=0.8,
    message="Budget alert: {cost} / {budget}"  # Template sanitized
)

# ❌ Rejected - HTTP not allowed
# alerts.add_webhook("http://example.com/hook", threshold=0.8)
```

### 7. File Export Security

Protection against path traversal and data exposure in exports:

- **Path Validation**: Prevents directory traversal attacks
- **File Extension Filtering**: Only safe export formats allowed
- **Data Size Limits**: Prevents large data exports (50K CSV, 10K JSON)
- **Filename Sanitization**: Removes dangerous characters
- **Working Directory Enforcement**: Files only created in safe locations

```python
from argentum import CostExporter

exporter = CostExporter()

# ✅ Safe - validates path and limits data size
exporter.export_csv("reports/costs.csv")

# ❌ Rejected - path traversal attempt
# exporter.export_csv("../../../etc/passwd")
```

## 🔧 Security Configuration

### Production Configuration

```python
from argentum.security import configure_security

# Recommended production settings
configure_security(
    max_state_size_mb=5,          # Conservative state size
    max_context_items=5000,       # Reasonable context limit
    max_plan_steps=500,           # Prevent overly complex plans
    enable_all_protections=True   # All security features enabled
)
```

### Development Configuration

```python
# More permissive for development
configure_security(
    max_state_size_mb=50,         # Larger for debugging
    max_context_items=50000,      # More context for testing
    max_plan_steps=2000,          # Complex test scenarios
    enable_all_protections=True   # Keep protections enabled
)
```

### Custom Security Policies

```python
from argentum.security import SecurityConfig, set_security_config

# Custom security configuration
custom_config = SecurityConfig(
    max_state_size=1024 * 1024,      # 1MB states
    max_context_items=1000,          # Limited context
    max_plan_steps=100,              # Simple plans only
    enable_xss_protection=True,      # XSS prevention
    enable_injection_protection=True, # Injection prevention
    sanitize_log_data=True           # Log sanitization
)

set_security_config(custom_config)
```

## 🚨 Security Best Practices

### 1. Validate All Inputs

```python
from argentum.security import sanitize_string, validate_json_size

# Sanitize user input
safe_input = sanitize_string(user_input, max_length=1000)

# Validate data size before processing
validate_json_size(large_data, max_size=10*1024*1024, context="user_data")
```

### 2. Use Built-in Secrets Detection

```python
# Always scan for secrets in agent plans
secrets_patterns = [
    "sk-",           # OpenAI keys
    "ghp_",          # GitHub tokens
    "api_key",       # Generic API keys
    "password",      # Passwords
    "secret",        # Generic secrets
    "bearer",        # Bearer tokens
]

result = linter.lint(plan, tools, secrets=secrets_patterns)
```

### 3. Implement Size Limits

```python
# Check state size before processing
import json
state_json = json.dumps(agent_state)
if len(state_json) > 10_000_000:  # 10MB limit
    raise ValueError("Agent state too large for safe processing")
```

### 4. Sanitize Logged Data

```python
# ❌ Don't log sensitive data
logger.info(f"Processing plan: {full_plan_data}")

# ✅ Log safely with IDs only
logger.info("Processing plan", extra={
    "plan_id": plan_id,
    "step_count": len(plan["steps"]),
    "estimated_time": estimate_time(plan)
})
```

### 5. Validate File Paths

```python
from argentum.security import validate_path

# Validate artifact paths
for artifact in handoff.artifacts:
    safe_path = validate_path(artifact, allow_relative=True)
    # Process safe_path...
```

### 6. Secure Webhook Configuration

```python
from argentum import CostAlerts

alerts = CostAlerts()

# ✅ Best practices for webhook security
alerts.add_slack_webhook(
    "https://hooks.slack.com/services/...",  # Always use HTTPS
    threshold=0.8,                          # Reasonable threshold
    channel="#ai-costs",                    # Specific channel
    username="Argentum-Bot"                 # Clear bot identity
)

# ✅ Safe email configuration
alerts.add_email(
    "finance@company.com",                  # Corporate email only
    threshold=1.0,                         # Conservative threshold
    subject="AI Cost Alert"                 # Short, clean subject
)
```

### 7. Safe Export Practices

```python
from argentum import CostExporter
from argentum.cost_export import ExportConfig

exporter = CostExporter(cost_tracker)

# ✅ Safe export configuration
config = ExportConfig(
    include_agent_breakdown=True,           # Business data only
    include_timestamps=False,              # Reduce data exposure
    date_format="%Y-%m-%d"                 # Simple date format
)

# Export to controlled locations
exporter.export_csv("./reports/costs.csv", config=config)

# ✅ Secure dashboard sharing
dashboard_url = exporter.generate_dashboard_url(
    DashboardConfig(
        public_access=False,               # Internal access only
        expiry_hours=24,                   # Short expiry time
        title="Cost Dashboard"             # Simple title
    )
)
```

## 🔍 Security Testing

### Automated Security Scans

The CI/CD pipeline includes:
- **Bandit**: Static security analysis
- **Safety**: Dependency vulnerability scanning  
- **Secrets Detection**: Custom patterns for credentials
- **Input Fuzzing**: Random input validation testing

### Manual Security Testing

```python
# Test with malicious inputs
malicious_inputs = [
    "<script>alert('xss')</script>",
    "'; DROP TABLE users; --",
    "../../../etc/passwd",
    "javascript:alert('xss')",
    "__import__('os').system('ls')"
]

for malicious in malicious_inputs:
    try:
        # Should safely reject or sanitize
        result = sanitize_string(malicious)
        print(f"Sanitized: {malicious} -> {result}")
    except SecurityError as e:
        print(f"Correctly rejected: {e}")
```

### Security Testing for New Features

```python
from argentum import CostAlerts, CostExporter
from argentum.cost_alerts import SecurityError
from argentum.cost_export import ExportSecurityError

# Test webhook security
alerts = CostAlerts()

# Test malicious webhook URLs
malicious_urls = [
    "http://evil.com/hook",              # HTTP should be rejected
    "https://localhost:8080/hook",       # Localhost should be rejected
    "https://192.168.1.1/hook",         # Private IP should be rejected
    "javascript:alert('xss')"            # Invalid scheme
]

for url in malicious_urls:
    try:
        alerts.add_webhook(url, threshold=0.8)
        print(f"⚠️  URL incorrectly accepted: {url}")
    except SecurityError:
        print(f"✅ URL correctly rejected: {url}")

# Test export path traversal
exporter = CostExporter()

malicious_paths = [
    "../../../etc/passwd",              # Path traversal
    "/tmp/../../secret.txt",            # Absolute with traversal  
    "file.exe",                         # Invalid extension
    "very_long_filename_" + "x"*300,    # Too long filename
    "file|with|pipes.csv"               # Dangerous characters
]

for path in malicious_paths:
    try:
        exporter.export_csv(path)
        print(f"⚠️  Path incorrectly accepted: {path}")
    except (ExportSecurityError, OSError):
        print(f"✅ Path correctly rejected: {path}")
```

## 📊 Security Monitoring

### Logging Security Events

```python
import logging
security_logger = logging.getLogger('argentum.security')

# Security events are automatically logged
# - Large input detection
# - Secrets detection
# - Validation failures
# - Suspicious patterns
```

### Metrics to Monitor

- Input size violations
- Secrets detection frequency
- Validation failure rates
- Resource usage patterns
- Error rates by security check

## 🚨 Vulnerability Reporting

### Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

### Reporting a Vulnerability

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please email security reports to: **[security@yourproject.com]**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested fixes (if any)

**Response Timeline:**
- **Initial Response**: Within 24 hours
- **Assessment**: Within 1 week  
- **Fix Release**: Within 2 weeks for critical issues

### Security Updates

Security updates will be:
- Released as patch versions (e.g., 0.1.1, 0.1.2)
- Clearly marked in release notes
- Announced through security advisories
- Backward compatible when possible

## 🔒 Security Checklist for Deployments

### Before Production Deployment

- [ ] Configure appropriate resource limits
- [ ] Enable all security protections
- [ ] Set up secrets detection patterns
- [ ] Configure secure logging
- [ ] Test with malicious inputs
- [ ] Review dependency vulnerabilities
- [ ] Enable security monitoring
- [ ] Document security configuration
- [ ] **Validate webhook URLs** for HTTPS and trusted domains
- [ ] **Test export path validation** with traversal attempts
- [ ] **Configure export size limits** appropriate for environment
- [ ] **Set dashboard expiry times** based on security policy
- [ ] **Review alert message templates** for injection risks

### Regular Security Maintenance

- [ ] Update dependencies regularly
- [ ] Review security logs
- [ ] Monitor resource usage
- [ ] Test security controls
- [ ] Update secrets patterns
- [ ] Review access controls
- [ ] Audit security configuration

## 📚 Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Guide](https://python-security.readthedocs.io/)
- [Secure Coding Practices](https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/)
- [Dependency Security](https://github.com/pyupio/safety)

---

**Remember**: Security is everyone's responsibility. When in doubt, choose the more secure option.