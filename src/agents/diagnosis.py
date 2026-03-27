"""Diagnosis Agent — uses RAG to analyze symptoms against the medical knowledge base."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from src.tools.rag import get_retriever


DIAGNOSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a medical diagnosis assistant. You analyze patient symptoms using a medical knowledge base.

Based on the retrieved medical knowledge and the patient's symptoms, provide your analysis in this EXACT format:

**Possible Conditions**: List the most likely conditions ranked by probability.

**Severity Level**: [Write exactly one: SELF_CARE, APPOINTMENT, URGENT, or EMERGENCY]

**Recommended Specialty**: [Write exactly one specialty if APPOINTMENT/URGENT: General Practice, Cardiology, Dermatology, Orthopedics, Pulmonology, ENT, Neurology, Gastroenterology, or Psychiatry. Write "None" if SELF_CARE or EMERGENCY]

**Recommended Action**:
- If SELF_CARE: Specify exact OTC medicines with dosages and self-care steps
- If APPOINTMENT: Explain why the patient needs to see the specialist above
- If EMERGENCY: Specify what emergency action to take

**Red Flags to Watch For**: Warning signs that would escalate severity

**Disclaimer**: This is AI-assisted guidance, not a professional medical diagnosis. Please consult a doctor.

Medical Knowledge Base Context:
{context}"""),
    ("human", "{symptoms}"),
])


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def create_diagnosis_chain(llm: ChatGroq):
    """Create the RAG-powered diagnosis chain."""
    retriever = get_retriever(k=6)

    chain = (
        {"context": retriever | format_docs, "symptoms": RunnablePassthrough()}
        | DIAGNOSIS_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain
