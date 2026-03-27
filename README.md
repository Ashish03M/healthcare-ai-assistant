# Healthcare AI Assistant

A multi-agent healthcare chatbot for patients in India, powered by **LangGraph**, **Groq (Llama 3.3 70B)**, and **RAG-based symptom diagnosis**.

## What It Does

1. **Symptom Assessment** - Asks follow-up questions to understand the patient condition (duration, severity, related symptoms)
2. **RAG-Powered Diagnosis** - Analyzes symptoms against a medical knowledge base using FAISS vector search and returns possible conditions, severity, and recommended actions
3. **Doctor Search and Booking** - Finds specialists across Delhi/NCR hospitals (Apollo, Max, Fortis, AIIMS, Medanta) and books appointments
4. **Email Confirmation** - Sends HTML appointment confirmation emails via SMTP

## Architecture

```
User (CLI) --> LangGraph ReAct Agent --> Tools
                                          |-- analyze_symptoms (RAG: FAISS + HuggingFace embeddings)
                                          |-- search_doctors (in-memory doctor pool)
                                          |-- get_available_slots
                                          |-- book_appointment
                                          +-- send_confirmation_email (SMTP)
```

**Tech Stack:**
- **LLM**: Groq (Llama 3.3 70B Versatile)
- **Agent Framework**: LangGraph `create_react_agent`
- **RAG**: FAISS + `all-MiniLM-L6-v2` embeddings
- **Knowledge Base**: Markdown files covering symptoms, conditions, and treatment guidelines for India
- **CLI**: Rich (panels, markdown rendering, spinners)

## Setup

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com)

### Installation

```bash
git clone https://github.com/Ashish03M/healthcare-ai-assistant.git
cd healthcare-ai-assistant
pip install -e .
cp .env.example .env
```

### Configuration

Edit `.env` with your credentials:

```env
# Required
GROQ_API_KEY=gsk_your_key_here

# Optional - for real email confirmations (leave empty for simulated emails)
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
SENDER_EMAIL=your_email@gmail.com
```

### Run

```bash
python main.py
```

## Example Conversation

```
You: I have a bad fever since 2 days and body pain
Assistant: I am sorry to hear that. Can you tell me more?
          - How high has your temperature been?
          - Any other symptoms like headache, sore throat, rash?
          - Have you taken any medicines?

You: Fever is around 102F, I have headache and sore throat too. Took Dolo 650.
Assistant: [Calls analyze_symptoms tool, gets RAG diagnosis]
          Based on your symptoms, this looks like a viral fever.
          Possible conditions: Viral Fever, Common Cold, Typhoid (less likely).
          Recommended: See a General Practice doctor. Would you like me to find one?

You: Yes please
Assistant: [Calls search_doctors tool]
          Here are available GPs in Delhi:
          1. Dr. Anjali Sharma - Apollo Clinic, Connaught Place
          2. Dr. Rajesh Verma - Max Multi Speciality, Panchsheel Park
```

## Project Structure

```
healthcare-ai-assistant/
|-- main.py                          # CLI entry point
|-- pyproject.toml                   # Project config and dependencies
|-- knowledge/                       # Medical knowledge base (RAG source)
|   |-- symptoms_conditions.md
|   |-- treatment_guidelines.md
|   +-- doctor_directory.md
+-- src/
    |-- config.py                    # Settings from .env
    |-- graph.py                     # LangGraph ReAct agent setup
    |-- state.py                     # Agent state definition
    |-- agents/
    |   |-- diagnosis.py             # RAG diagnosis chain
    |   +-- appointment.py           # Appointment agent
    +-- tools/
        |-- rag.py                   # FAISS vector store and retriever
        |-- doctor_pool.py           # Doctor search, slots, booking
        +-- email_service.py         # SMTP email confirmations
```

## Specialties Covered

| Specialty | Doctor | Hospital |
|-----------|--------|----------|
| General Practice | Dr. Anjali Sharma | Apollo Clinic, New Delhi |
| General Practice | Dr. Rajesh Verma | Max Multi Speciality, New Delhi |
| Cardiology | Dr. Priya Nair | Fortis Escorts Heart Institute |
| Dermatology | Dr. Vikram Mehta | AIIMS, New Delhi |
| Orthopedics | Dr. Suresh Reddy | Medanta, Gurugram |
| Pulmonology | Dr. Kavita Iyer | Sir Ganga Ram Hospital |
| ENT | Dr. Arun Joshi | BLK-Max Hospital |
| Neurology | Dr. Deepak Gupta | AIIMS Neurosciences Centre |
| Gastroenterology | Dr. Meena Krishnan | Indraprastha Apollo Hospital |
| Psychiatry | Dr. Shalini Das | NIMHANS Outreach Centre |

## Disclaimer

This is an AI-assisted healthcare assistant for **demonstration purposes**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.
