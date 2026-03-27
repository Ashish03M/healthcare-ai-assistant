"""Shared state definition for the multi-agent healthcare workflow."""

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared state passed between all nodes in the graph."""
    messages: Annotated[list, add_messages]
    diagnosis_result: str
    severity: Literal["SELF_CARE", "APPOINTMENT", "URGENT", "EMERGENCY", ""]
    recommended_specialty: str
    current_agent: str
    # Appointment workflow phases
    appointment_phase: Literal["none", "awaiting_choice", "awaiting_details", "booked", ""]
    selected_doctor_id: str
    selected_slot: str
    patient_name: str
    patient_age: str
    patient_email: str
