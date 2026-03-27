"""Appointment Agent — multi-turn stateful appointment booking."""

import json
import re
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.tools.doctor_pool import search_doctors, get_available_slots, book_appointment
from src.tools.email_service import send_confirmation_email


def search_and_present_doctors(specialty: str) -> tuple[str, list[dict]]:
    """Search for doctors and return a formatted presentation + raw data."""
    result = search_doctors.invoke({"specialty": specialty})
    doctors = json.loads(result)

    if not doctors:
        return f"I couldn't find any {specialty} specialists available. Let me search General Practice instead.", []

    lines = [f"I found the following **{specialty}** specialists available for you:\n"]
    for i, doc in enumerate(doctors, 1):
        lines.append(f"**{i}. {doc['name']}**")
        lines.append(f"   - Next available: {doc['next_available_date']}")
        lines.append(f"   - Time slots: {', '.join(doc['available_slots'])}")
        lines.append(f"   - Location: {doc['location']}")
        lines.append("")

    lines.append("Please tell me:")
    lines.append("1. Which doctor you'd prefer (name or number)")
    lines.append("2. Which time slot works for you")
    lines.append("3. Your **full name**, **age**, and **email address** so I can book and send you a confirmation")
    lines.append("\nFor example: *\"Dr. Rodriguez at 9:30 AM, my name is John Smith, age 34, email john@email.com\"*")

    return "\n".join(lines), doctors


def parse_patient_details(text: str, doctors: list[dict] = None) -> dict:
    """Try to extract doctor choice, slot, name, age, and email from user text."""
    info = {}

    # Parse email
    email_match = re.search(r'[\w.+-]+@[\w-]+\.[\w.]+', text)
    if email_match:
        info["patient_email"] = email_match.group(0)

    # Parse age — "age 33", "33 years old", or standalone number near name/email context
    age_match = re.search(r'(?:age\s*[:\s]?\s*|,\s*age\s+)(\d{1,3})\b', text, re.IGNORECASE)
    if not age_match:
        age_match = re.search(r'\b(\d{1,3})\s*(?:years?\s*old|yrs?\s*old|y/?o)\b', text, re.IGNORECASE)
    if not age_match:
        # Standalone 2-digit number not part of a time (not followed by :, AM/PM, or preceded by "at")
        age_match = re.search(r'(?<!\d[:\.])\b(\d{2})\b(?!\s*(?:am|pm|:|/))', text, re.IGNORECASE)
    if age_match:
        age = int(age_match.group(1))
        if 10 <= age <= 120:
            info["patient_age"] = str(age)

    # Parse name — multiple patterns:
    # "my name is X Y", "I'm X Y", "name: X", or just a capitalized word(s) before age/email
    name_match = re.search(
        r"(?:(?:my\s+)?name\s+is|i'?m|i\s+am)\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)",
        text, re.IGNORECASE,
    )
    if not name_match:
        # Try "...Name Age Email..." pattern — word(s) before a number or @
        name_match = re.search(
            r'(?:^|[.!,]\s*)([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)*)\s+\d{2}\b',
            text,
        )
    if not name_match:
        # Try word(s) just before an email address
        name_match = re.search(
            r'(?:^|[.!,]\s*)([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)*)\s+[\w.+-]+@',
            text,
        )
    if name_match:
        candidate = name_match.group(1).strip()
        # Filter out words that are clearly not names
        skip_words = {"dr", "doctor", "at", "am", "pm", "friday", "monday", "tuesday",
                      "wednesday", "thursday", "saturday", "sunday", "please", "the", "my"}
        words = candidate.split()
        clean = [w for w in words if w.lower() not in skip_words]
        if clean:
            info["patient_name"] = " ".join(w.capitalize() for w in clean)

    # Parse doctor choice — supports partial matching like "dr sarah" or "sarah"
    if doctors:
        text_lower = text.lower()
        # Exact full name match first
        for doc in doctors:
            if doc["name"].lower() in text_lower or doc["id"] in text_lower:
                info["selected_doctor_id"] = doc["id"]
                info["doctor_name"] = doc["name"]
                info["doctor_location"] = doc["location"]
                info["doctor_specialty"] = doc["specialty"]
                break
        # Partial match — match any part of doctor's name (first or last name)
        if "selected_doctor_id" not in info:
            for doc in doctors:
                name_parts = doc["name"].lower().replace("dr.", "").replace("dr", "").split()
                for part in name_parts:
                    if part and len(part) > 2 and part in text_lower:
                        info["selected_doctor_id"] = doc["id"]
                        info["doctor_name"] = doc["name"]
                        info["doctor_location"] = doc["location"]
                        info["doctor_specialty"] = doc["specialty"]
                        break
                if "selected_doctor_id" in info:
                    break
        # Number reference ("doctor 1", "#1", "1.")
        if "selected_doctor_id" not in info:
            num_match = re.search(r'(?:doctor\s*|#|option\s*)(\d)', text, re.IGNORECASE)
            if num_match:
                idx = int(num_match.group(1)) - 1
                if 0 <= idx < len(doctors):
                    doc = doctors[idx]
                    info["selected_doctor_id"] = doc["id"]
                    info["doctor_name"] = doc["name"]
                    info["doctor_location"] = doc["location"]
                    info["doctor_specialty"] = doc["specialty"]

    # Parse time slot — "9:00 AM", "9 am", "9am", "2:30 PM"
    time_match = re.search(r'\b(\d{1,2}:\d{2})\s*(AM|PM|am|pm)\b', text, re.IGNORECASE)
    if time_match:
        info["selected_slot"] = f"{time_match.group(1)} {time_match.group(2).upper()}"
    else:
        # Handle "9 am", "9am", "2 pm" — convert to "9:00 AM"
        time_match = re.search(r'\b(\d{1,2})\s*(am|pm)\b', text, re.IGNORECASE)
        if time_match:
            hour = time_match.group(1)
            period = time_match.group(2).upper()
            info["selected_slot"] = f"{hour}:00 {period}"

    return info


def do_booking_and_email(
    doctor_id: str, slot: str, patient_name: str, patient_age: str, patient_email: str,
) -> str:
    """Book the appointment and send the confirmation email. Returns formatted result."""
    # Book
    booking_result = book_appointment.invoke({
        "doctor_id": doctor_id,
        "slot": slot,
        "patient_name": patient_name,
        "patient_age": int(patient_age),
        "patient_email": patient_email,
    })
    booking = json.loads(booking_result)

    if booking.get("status") != "confirmed":
        return f"Sorry, there was an issue booking: {booking_result}"

    # Send email
    email_result = send_confirmation_email.invoke({
        "patient_email": patient_email,
        "patient_name": patient_name,
        "doctor_name": booking["doctor"],
        "specialty": booking["specialty"],
        "date": booking["date"],
        "time": booking["time"],
        "location": booking["location"],
        "booking_id": booking["booking_id"],
    })
    email_info = json.loads(email_result)

    lines = [
        "Your appointment has been **confirmed**! Here are the details:\n",
        f"- **Booking ID**: {booking['booking_id']}",
        f"- **Doctor**: {booking['doctor']} ({booking['specialty']})",
        f"- **Date**: {booking['date']}",
        f"- **Time**: {booking['time']}",
        f"- **Location**: {booking['location']}",
        f"- **Patient**: {patient_name}, Age {patient_age}",
        "",
    ]

    if email_info.get("status") == "sent":
        lines.append(f"A confirmation email has been sent to **{patient_email}**.")
    elif email_info.get("status") == "simulated":
        lines.append(f"[Email] {email_info['message']}")
    else:
        lines.append(f"Note: {email_info.get('message', 'Email could not be sent, but your appointment is confirmed.')}")

    lines.append("\nPlease arrive 15 minutes before your appointment time. Is there anything else I can help you with?")

    return "\n".join(lines)
