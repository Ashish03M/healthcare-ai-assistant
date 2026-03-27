"""Email service for sending appointment confirmations."""

import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain_core.tools import tool

from src.config import Settings


def _build_html_email(
    patient_name: str,
    doctor_name: str,
    specialty: str,
    date: str,
    time: str,
    location: str,
    booking_id: str,
) -> str:
    return f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <div style="background-color: #2563eb; color: white; padding: 20px; text-align: center;">
            <h1 style="margin: 0;">Appointment Confirmed</h1>
        </div>
        <div style="padding: 20px; background-color: #f8fafc;">
            <p>Dear <strong>{patient_name}</strong>,</p>
            <p>Your appointment has been successfully booked. Here are the details:</p>
            <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                <tr style="border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 10px; font-weight: bold;">Booking ID</td>
                    <td style="padding: 10px;">{booking_id}</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 10px; font-weight: bold;">Doctor</td>
                    <td style="padding: 10px;">{doctor_name}</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 10px; font-weight: bold;">Specialty</td>
                    <td style="padding: 10px;">{specialty}</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 10px; font-weight: bold;">Date</td>
                    <td style="padding: 10px;">{date}</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 10px; font-weight: bold;">Time</td>
                    <td style="padding: 10px;">{time}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; font-weight: bold;">Location</td>
                    <td style="padding: 10px;">{location}</td>
                </tr>
            </table>
            <p>Please arrive 15 minutes before your scheduled time.</p>
            <p>If you need to reschedule or cancel, please contact us at least 24 hours in advance.</p>
            <hr style="border: none; border-top: 1px solid #e2e8f0;">
            <p style="color: #64748b; font-size: 12px;">
                This is an automated message from Healthcare AI Assistant.
                Please do not reply to this email.
            </p>
        </div>
    </body>
    </html>
    """


@tool
def send_confirmation_email(
    patient_email: str,
    patient_name: str,
    doctor_name: str,
    specialty: str,
    date: str,
    time: str,
    location: str,
    booking_id: str,
) -> str:
    """Send an appointment confirmation email to the patient.

    Args:
        patient_email: Patient's email address
        patient_name: Patient's full name
        doctor_name: Doctor's full name
        specialty: Doctor's specialty
        date: Appointment date
        time: Appointment time
        location: Clinic/hospital location
        booking_id: Unique booking reference ID
    """
    try:
        settings = Settings.from_env()

        if not settings.smtp_user or not settings.smtp_password:
            return json.dumps({
                "status": "simulated",
                "message": f"Email confirmation would be sent to {patient_email}. (SMTP not configured - set SMTP_USER and SMTP_PASSWORD in .env to enable real emails)",
                "recipient": patient_email,
                "booking_id": booking_id,
            })

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Appointment Confirmed - {doctor_name} on {date} at {time}"
        msg["From"] = settings.sender_email
        msg["To"] = patient_email

        html_body = _build_html_email(
            patient_name=patient_name,
            doctor_name=doctor_name,
            specialty=specialty,
            date=date,
            time=time,
            location=location,
            booking_id=booking_id,
        )
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.starttls()
            server.login(settings.smtp_user, settings.smtp_password)
            server.send_message(msg)

        return json.dumps({
            "status": "sent",
            "message": f"Confirmation email sent to {patient_email}",
            "recipient": patient_email,
            "booking_id": booking_id,
        })

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to send email: {str(e)}. The appointment is still confirmed (Booking ID: {booking_id}).",
            "recipient": patient_email,
            "booking_id": booking_id,
        })
