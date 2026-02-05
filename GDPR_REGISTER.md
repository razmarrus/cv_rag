# GDPR Data Processing Register

## Processing Activity: RAG System Query Logging

**Data Controller:** Margot Razumeyeva
**Contact:** margo.razumeyeva@gmail.com
**Location:** Amsterdam, Netherlands

### Personal Data Processed
- IP addresses (personal data under GDPR Art. 4(1))
- User questions (may contain personal data)
- System-generated answers
- Timestamps

### Purpose of Processing
- Rate limiting to prevent abuse
- Service improvement and debugging
- Usage analytics

### Legal Basis
Legitimate interest (GDPR Art. 6(1)(f))
- Interest: Ensuring service availability and preventing abuse
- Balancing test: Minimal data collected, short retention, users informed

### Data Sources
- Directly from users (HTTP requests)
- System-generated (answers, timestamps)

### Data Recipients
- No third-party sharing
- HuggingFace API (question/context for LLM processing - see their DPA)

### Data Transfers
- Data stored in EU (Netherlands)
- HuggingFace processing (check their data location)

### Retention Period
90 days from collection, then automatic deletion

### Technical and Organizational Measures
- Database encryption at rest
- TLS for data in transit
- Access controls (only data controller)
- Automated deletion after 90 days
- Regular security updates

### Data Subject Rights
Users can request:
- Access to their data
- Rectification (correction)
- Erasure (deletion)
- Restriction of processing
- Objection to processing
- Data portability

Requests handled via: margo.razumeyeva@gmail.com
Response time: Within 30 days (GDPR requirement)