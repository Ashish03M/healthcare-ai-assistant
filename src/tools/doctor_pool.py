"""Doctor pool management — search, availability, and booking functions (India-centric)."""

import json
from datetime import datetime, timedelta
from langchain_core.tools import tool

DOCTORS = {
    "doc_001": {
        "id": "doc_001",
        "name": "Dr. Anjali Sharma",
        "specialty": "General Practice",
        "available_days": ["Monday", "Wednesday", "Friday"],
        "slots": ["9:00 AM", "10:00 AM", "11:00 AM", "2:00 PM", "3:00 PM", "4:00 PM"],
        "location": "Apollo Clinic, Connaught Place, New Delhi",
        "email": "dr.sharma@apolloclinic.example.com",
    },
    "doc_002": {
        "id": "doc_002",
        "name": "Dr. Rajesh Verma",
        "specialty": "General Practice",
        "available_days": ["Tuesday", "Thursday", "Saturday"],
        "slots": ["8:30 AM", "10:00 AM", "11:30 AM", "1:00 PM", "2:30 PM"],
        "location": "Max Multi Speciality Centre, Panchsheel Park, New Delhi",
        "email": "dr.verma@maxhealthcare.example.com",
    },
    "doc_003": {
        "id": "doc_003",
        "name": "Dr. Priya Nair",
        "specialty": "Cardiology",
        "available_days": ["Monday", "Tuesday", "Thursday"],
        "slots": ["9:00 AM", "11:00 AM", "2:00 PM", "4:00 PM"],
        "location": "Fortis Escorts Heart Institute, Okhla Road, New Delhi",
        "email": "dr.nair@fortis.example.com",
    },
    "doc_004": {
        "id": "doc_004",
        "name": "Dr. Vikram Mehta",
        "specialty": "Dermatology",
        "available_days": ["Monday", "Wednesday", "Friday"],
        "slots": ["10:00 AM", "11:30 AM", "1:00 PM", "3:00 PM"],
        "location": "AIIMS Dermatology OPD, Ansari Nagar, New Delhi",
        "email": "dr.mehta@aiims.example.com",
    },
    "doc_005": {
        "id": "doc_005",
        "name": "Dr. Suresh Reddy",
        "specialty": "Orthopedics",
        "available_days": ["Tuesday", "Wednesday", "Thursday"],
        "slots": ["8:00 AM", "9:30 AM", "11:00 AM", "1:30 PM", "3:00 PM"],
        "location": "Medanta - The Medicity, Sector 38, Gurugram",
        "email": "dr.reddy@medanta.example.com",
    },
    "doc_006": {
        "id": "doc_006",
        "name": "Dr. Kavita Iyer",
        "specialty": "Pulmonology",
        "available_days": ["Monday", "Thursday", "Friday"],
        "slots": ["9:00 AM", "10:30 AM", "12:00 PM", "2:00 PM", "3:30 PM"],
        "location": "Sir Ganga Ram Hospital, Rajinder Nagar, New Delhi",
        "email": "dr.iyer@gangaram.example.com",
    },
    "doc_007": {
        "id": "doc_007",
        "name": "Dr. Arun Joshi",
        "specialty": "ENT",
        "available_days": ["Tuesday", "Wednesday", "Friday"],
        "slots": ["9:30 AM", "11:00 AM", "1:00 PM", "2:30 PM", "4:00 PM"],
        "location": "BLK-Max Super Speciality Hospital, Pusa Road, New Delhi",
        "email": "dr.joshi@blkmax.example.com",
    },
    "doc_008": {
        "id": "doc_008",
        "name": "Dr. Deepak Gupta",
        "specialty": "Neurology",
        "available_days": ["Monday", "Wednesday", "Thursday"],
        "slots": ["9:00 AM", "10:30 AM", "12:00 PM", "2:00 PM", "3:30 PM"],
        "location": "AIIMS Neurosciences Centre, Ansari Nagar, New Delhi",
        "email": "dr.gupta@aiims.example.com",
    },
    "doc_009": {
        "id": "doc_009",
        "name": "Dr. Meena Krishnan",
        "specialty": "Gastroenterology",
        "available_days": ["Monday", "Tuesday", "Thursday"],
        "slots": ["8:30 AM", "10:00 AM", "11:30 AM", "1:30 PM", "3:00 PM"],
        "location": "Indraprastha Apollo Hospital, Sarita Vihar, New Delhi",
        "email": "dr.krishnan@apollohospital.example.com",
    },
    "doc_010": {
        "id": "doc_010",
        "name": "Dr. Shalini Das",
        "specialty": "Psychiatry",
        "available_days": ["Tuesday", "Thursday", "Friday"],
        "slots": ["9:00 AM", "10:30 AM", "12:00 PM", "2:00 PM", "3:30 PM"],
        "location": "NIMHANS Outreach Centre, Lajpat Nagar, New Delhi",
        "email": "dr.das@nimhans.example.com",
    },
}

# In-memory booking store
BOOKINGS: list[dict] = []


def _next_available_date(available_days: list[str]) -> str:
    """Find the next date that falls on one of the available days."""
    day_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6,
    }
    today = datetime.now()
    for offset in range(1, 14):
        candidate = today + timedelta(days=offset)
        day_name = candidate.strftime("%A")
        if day_name in available_days:
            return candidate.strftime("%A, %B %d, %Y")
    return "No available date in the next 2 weeks"


@tool
def search_doctors(specialty: str) -> str:
    """Search for available doctors by specialty. Returns a list of matching doctors with their details.

    Args:
        specialty: The medical specialty to search for (e.g., 'General Practice', 'Cardiology', 'Dermatology', 'Orthopedics', 'Pulmonology', 'ENT', 'Neurology', 'Gastroenterology', 'Psychiatry')
    """
    specialty_lower = specialty.lower()
    matches = []
    for doc in DOCTORS.values():
        if specialty_lower in doc["specialty"].lower():
            next_date = _next_available_date(doc["available_days"])
            matches.append({
                "id": doc["id"],
                "name": doc["name"],
                "specialty": doc["specialty"],
                "next_available_date": next_date,
                "available_slots": doc["slots"],
                "location": doc["location"],
            })

    if not matches:
        return f"No doctors found for specialty: {specialty}. Available specialties: General Practice, Cardiology, Dermatology, Orthopedics, Pulmonology, ENT, Neurology, Gastroenterology, Psychiatry."

    return json.dumps(matches, indent=2)


@tool
def get_available_slots(doctor_id: str) -> str:
    """Get available appointment slots for a specific doctor.

    Args:
        doctor_id: The doctor's ID (e.g., 'doc_001')
    """
    doc = DOCTORS.get(doctor_id)
    if not doc:
        return f"Doctor with ID {doctor_id} not found."

    next_date = _next_available_date(doc["available_days"])

    booked_slots = {
        b["slot"] for b in BOOKINGS
        if b["doctor_id"] == doctor_id and b["date"] == next_date
    }
    available = [s for s in doc["slots"] if s not in booked_slots]

    return json.dumps({
        "doctor": doc["name"],
        "date": next_date,
        "available_slots": available,
        "location": doc["location"],
    }, indent=2)


@tool
def book_appointment(
    doctor_id: str,
    slot: str,
    patient_name: str,
    patient_age: int,
    patient_email: str,
) -> str:
    """Book an appointment with a doctor. Returns booking confirmation details.

    Args:
        doctor_id: The doctor's ID (e.g., 'doc_001')
        slot: The time slot (e.g., '10:00 AM')
        patient_name: The patient's full name
        patient_age: The patient's age
        patient_email: The patient's email address for confirmation
    """
    doc = DOCTORS.get(doctor_id)
    if not doc:
        return f"Doctor with ID {doctor_id} not found."

    next_date = _next_available_date(doc["available_days"])

    if slot not in doc["slots"]:
        return f"Slot {slot} is not available for {doc['name']}. Available slots: {', '.join(doc['slots'])}"

    booking = {
        "booking_id": f"BK-{len(BOOKINGS) + 1001}",
        "doctor_id": doctor_id,
        "doctor_name": doc["name"],
        "doctor_email": doc["email"],
        "specialty": doc["specialty"],
        "date": next_date,
        "slot": slot,
        "location": doc["location"],
        "patient_name": patient_name,
        "patient_age": patient_age,
        "patient_email": patient_email,
    }
    BOOKINGS.append(booking)

    return json.dumps({
        "status": "confirmed",
        "booking_id": booking["booking_id"],
        "doctor": doc["name"],
        "specialty": doc["specialty"],
        "date": next_date,
        "time": slot,
        "location": doc["location"],
        "patient_name": patient_name,
        "patient_email": patient_email,
    }, indent=2)
