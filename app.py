from flask import Flask, request, jsonify, send_file, render_template
import re
import uuid

app = Flask(__name__)
import requests
import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
from langchain.agents import initialize_agent, AgentType
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.units import inch
import uuid
import re
import qrcode
from reportlab.platypus import Image

load_dotenv()
TRIP_MEMORY = {
    "source": None,
    "destination": None,
    "start_date": None,
    "end_date": None,
    "flights": None,
    "selected_flight": None,
    "hotels": None,
    "selected_hotel": None,
    "itinerary": None,

    "user_name": None,         
    "user_age": None,
    "adults": 1,
    "traveler_names": [],   
    "total_cost": None,
    "payment_status": None,

    "step": 1
}

AIRLINE_CODE_MAP = {
    "AI": "Air India",
    "6E": "IndiGo",
    "UK": "Vistara",
    "SG": "SpiceJet",
    "G8": "Go First",
    "IX": "Air India Express"
}

API_KEY = os.getenv("AMADEUS_API_KEY")
API_SECRET = os.getenv("AMADEUS_API_SECRET")

TOKEN_URL = "https://test.api.amadeus.com/v1/security/oauth2/token"
HOTEL_LIST_URL = "https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city"
FLIGHT_URL = "https://test.api.amadeus.com/v2/shopping/flight-offers"


def get_access_token():
    data = {
        "grant_type": "client_credentials",
        "client_id": API_KEY,
        "client_secret": API_SECRET
    }

    response = requests.post(TOKEN_URL, data=data)
    return response.json()["access_token"]
llm1 = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7
)
def calculate_nights(start_date: str, end_date: str) -> int:
    s = datetime.strptime(start_date, "%Y-%m-%d")
    e = datetime.strptime(end_date, "%Y-%m-%d")
    return max((e - s).days, 1)


def calculate_total_cost():
    nights = calculate_nights(TRIP_MEMORY["start_date"], TRIP_MEMORY["end_date"])
    flight_price = float(TRIP_MEMORY["selected_flight"]["price"])
    hotel_price_per_night = float(TRIP_MEMORY["selected_hotel"]["price"])
    adults = int(TRIP_MEMORY["adults"])

    flight_cost = flight_price * adults         
    hotel_cost = hotel_price_per_night * nights

    TRIP_MEMORY["total_cost"] = flight_cost + hotel_cost

    return {
        "nights": nights,
        "flight_cost": flight_cost,
        "hotel_cost": hotel_cost,
        "total_cost": TRIP_MEMORY["total_cost"]
    }
def generate_qr(data: str, filename: str):
    """Generate a QR code image for given data."""
    qr = qrcode.make(data)
    qr.save(filename)
    return filename

def generate_booking_pdf(filename="trip_booking.pdf"):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(filename, pagesize=A4)
    elements = []

    elements.append(Paragraph("<b>TRIP BOOKING CONFIRMATION</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(
        f"Route: {TRIP_MEMORY['source']} → {TRIP_MEMORY['destination']}", styles["Normal"]
    ))
    elements.append(Paragraph(
        f"Trip Dates: {TRIP_MEMORY['start_date']} to {TRIP_MEMORY['end_date']}", styles["Normal"]
    ))
    elements.append(Paragraph(
        f"Number of Travelers: {TRIP_MEMORY['adults']}", styles["Normal"]
    ))
    elements.append(Spacer(1, 0.3 * inch))

    # ✅ FLIGHT TICKETS WITH QR CODES
    elements.append(Paragraph("<b>FLIGHT TICKETS</b>", styles["Heading2"]))

    qr_files = []

    for i, name in enumerate(TRIP_MEMORY["traveler_names"], 1):
        ticket_id = f"FL-{uuid.uuid4().hex[:6].upper()}"

        # ✅ QR DATA CONTENT
        qr_data = (
            f"Ticket ID: {ticket_id}\n"
            f"Passenger: {name}\n"
            f"Airline: {TRIP_MEMORY['selected_flight']['airline']}\n"
            f"Flight No: {TRIP_MEMORY['selected_flight']['flight_number']}\n"
            f"From: {TRIP_MEMORY['selected_flight']['departure_airport']}\n"
            f"To: {TRIP_MEMORY['selected_flight']['arrival_airport']}\n"
            f"Date: {TRIP_MEMORY['start_date']}"
        )

        qr_filename = f"qr_{ticket_id}.png"
        generate_qr(qr_data, qr_filename)
        qr_files.append(qr_filename)

        # Ticket table
        data = [
            ["Passenger Name", name],
            ["Ticket ID", ticket_id],
            ["Airline", TRIP_MEMORY["selected_flight"]["airline"]],
            ["Flight No", TRIP_MEMORY["selected_flight"]["flight_number"]],
            ["From", TRIP_MEMORY["selected_flight"]["departure_airport"]],
            ["To", TRIP_MEMORY["selected_flight"]["arrival_airport"]],
            ["Departure", TRIP_MEMORY["selected_flight"]["departure_time"]],
            ["Arrival", TRIP_MEMORY["selected_flight"]["arrival_time"]],
            ["Price", f"₹{TRIP_MEMORY['selected_flight']['price']}"],
        ]

        table = Table(data, colWidths=[2.5 * inch, 3.5 * inch])

        elements.append(table)
        elements.append(Spacer(1, 0.2 * inch))

        # ✅ Embed QR image
        qr_img = Image(qr_filename, width=1.5 * inch, height=1.5 * inch)
        elements.append(qr_img)
        elements.append(Spacer(1, 0.4 * inch))

    # ✅ HOTEL SECTION
    elements.append(Paragraph("<b>HOTEL BOOKING</b>", styles["Heading2"]))
    hotel_data = [
        ["Hotel", TRIP_MEMORY["selected_hotel"]["hotel_name"]],
        ["City", TRIP_MEMORY["selected_hotel"]["city"]],
        ["Check-in", TRIP_MEMORY["start_date"]],
        ["Check-out", TRIP_MEMORY["end_date"]],
        ["Price / Night", f"₹{TRIP_MEMORY['selected_hotel']['price']}"],
    ]

    elements.append(Table(hotel_data, colWidths=[2.5 * inch, 3.5 * inch]))
    elements.append(Spacer(1, 0.3 * inch))

    # ✅ ITINERARY
    elements.append(Paragraph("<b>ITINERARY</b>", styles["Heading2"]))
    for line in TRIP_MEMORY["itinerary"].split("\n"):
        elements.append(Paragraph(line, styles["Normal"]))

    elements.append(Spacer(1, 0.3 * inch))

    # ✅ PAYMENT SUMMARY
    elements.append(Paragraph("<b>PAYMENT SUMMARY</b>", styles["Heading2"]))
    elements.append(Paragraph(f"Total Paid: ₹{TRIP_MEMORY['total_cost']}", styles["Normal"]))
    elements.append(Paragraph("Payment Status: PAID", styles["Normal"]))

    # ✅ Build PDF
    doc.build(elements)

    # ✅ Cleanup QR images
    for f in qr_files:
        if os.path.exists(f):
            os.remove(f)

    return filename


@tool
def generate_trip_plan(start_date: str, end_date: str, source: str, destination: str) -> str:
    """
    ONLY use this tool AFTER both flight and hotel are selected.
    Generates a detailed day-wise travel plan using Gemini.
    
    """

   
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    total_days = (end - start).days + 1

    prompt = f"""
    Create a detailed {total_days}-day travel itinerary.

    Start Date: {start_date}
    End Date: {end_date}
    Source: {source}
    Destination: {destination}

    Rules:
    - Provide day-wise morning, afternoon, and evening plans.
    - Suggest famous attractions and local food.
    - Include travel tips.
    - Keep the plan practical.
    - Add one light/rest day if the trip is longer than 3 days.
    - Output only clean numbered format.
    """

    response = llm1.invoke(prompt)
    return response.content
CITY_CODE_MAP = {
    "delhi": "DEL",
    "new delhi": "DEL",
    "mumbai": "BOM",
    "bombay": "BOM",
    "bangalore": "BLR",
    "bengaluru": "BLR",
    "chennai": "MAA",
    "kolkata": "CCU"
}

def normalize_city(city):
    return CITY_CODE_MAP.get(city.lower(), city.upper())
@tool
def search_hotel(city: str, check_in: str, check_out: str):
    """Search hotels by city and return hotel names with demo prices"""

 
    city = normalize_city(city)

    token = get_access_token()

    headers = {
        "Authorization": f"Bearer {token}"
    }

    params = {
        "cityCode": city
    }

    response = requests.get(
        HOTEL_LIST_URL,
        headers=headers,
        params=params
    ).json()

    if "data" not in response or len(response["data"]) == 0:
        return {"error": f"No hotels found for city code {city}"}

    dummy_hotels = []
    base_price = 3500  

    for i, hotel in enumerate(response["data"][:5]):
        dummy_hotels.append({
            "hotel_name": hotel["name"],
            "city": hotel["address"].get("cityName", ""),
            "price": base_price + (i * 700),
            "rating": 4
        })

    return dummy_hotels
BOOKED_FLIGHTS = []
BOOKED_HOTELS = []


@tool
def book_flight(
    airline: str,
    flight_number: str,
    departure: str,
    arrival: str,
    price: float,
    currency: str,
    passenger_name: str
):
    """
    Dummy-book a flight ticket with the given flight details and passenger name.
    Returns a fake booking reference and stored booking details.
    """

    booking_id = "FL-" + str(uuid.uuid4())[:8]

    booking = {
        "booking_id": booking_id,
        "airline": airline,
        "flight_number": flight_number,
        "departure": departure,
        "arrival": arrival,
        "price": price,
        "currency": currency,
        "passenger_name": passenger_name,
        "status": "CONFIRMED"
    }

    BOOKED_FLIGHTS.append(booking)
    return booking


@tool
def book_hotel(
    hotel_name: str,
    city: str,
    price: float,
    check_in: str,
    check_out: str,
    guest_name: str):
    """
    Dummy-book a hotel room with the given hotel details and guest name.
    Returns a fake booking reference and stored booking details.
    """

    booking_id = "HT-" + str(uuid.uuid4())[:8]

    booking = {
        "booking_id": booking_id,
        "hotel_name": hotel_name,
        "city": city,
        "price": price,
        "check_in": check_in,
        "check_out": check_out,
        "guest_name": guest_name,
        "status": "CONFIRMED"
    }

    BOOKED_HOTELS.append(booking)
    return booking
@tool
def search_flight(origin: str, destination: str, date: str):
    """Search flights between two cities for a given date"""

    origin = normalize_city(origin)
    destination = normalize_city(destination)

    token = get_access_token()

    headers = {
        "Authorization": f"Bearer {token}"
    }

    params = {
        "originLocationCode": origin,
        "destinationLocationCode": destination,
        "departureDate": date,
        "adults": 1,
        "max": 5
    }

    response = requests.get(
        FLIGHT_URL,
        headers=headers,
        params=params
    ).json()

    if "data" not in response or len(response["data"]) == 0:
        return {"error": "No flights found"}

    flights = []

    for flight in response["data"][:5]:
        itinerary = flight["itineraries"][0]["segments"][0]

        carrier_code = itinerary["carrierCode"]
        airline_name = AIRLINE_CODE_MAP.get(carrier_code, carrier_code)

        flights.append({
            "airline": airline_name,  
            "flight_number": itinerary["number"],
            "departure_airport": itinerary["departure"]["iataCode"],
            "arrival_airport": itinerary["arrival"]["iataCode"],
            "departure_time": itinerary["departure"]["at"],
            "arrival_time": itinerary["arrival"]["at"],
            "price": round(float(flight["price"]["total"]) * 103.44, 2),
            "currency": "INR"
        })

    return flights

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.6
)

tools = [search_flight, search_hotel, generate_trip_plan,book_flight,book_hotel]

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    global TRIP_MEMORY

    user_input = request.json.get("message").strip()

    if user_input.lower() in ["exit", "quit"]:
        return jsonify({"reply": "Trip session ended."})

    # ---------------- STEP 1 ----------------
    if TRIP_MEMORY["step"] == 1:

        date_pattern = r"\d{4}-\d{2}-\d{2}"
        dates = re.findall(date_pattern, user_input)

        route_pattern = r"from\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+?)\s+from"
        route_match = re.search(route_pattern, user_input.lower())

        if len(dates) == 2 and route_match:
            TRIP_MEMORY["start_date"] = dates[0]
            TRIP_MEMORY["end_date"] = dates[1]
            TRIP_MEMORY["source"] = route_match.group(1).strip().title()
            TRIP_MEMORY["destination"] = route_match.group(2).strip().title()

            reply = (
                f"Trip Created ✅\n"
                f"From: {TRIP_MEMORY['source']}\n"
                f"To: {TRIP_MEMORY['destination']}\n"
                f"Dates: {TRIP_MEMORY['start_date']} → {TRIP_MEMORY['end_date']}\n\n"
                "How many adults are travelling?"
            )

            TRIP_MEMORY["step"] = "ask_adults"
            return jsonify({"reply": reply})

        return jsonify({"reply": "Please use format:\nCreate a trip from Delhi to Mumbai from 2025-12-10 to 2025-12-15"})

    # ---------------- ASK ADULTS ----------------
    if TRIP_MEMORY["step"] == "ask_adults":
        TRIP_MEMORY["adults"] = int(user_input)
        TRIP_MEMORY["traveler_names"] = []
        TRIP_MEMORY["step"] = "ask_names"
        return jsonify({"reply": "Enter name of traveler 1"})

    # ---------------- ASK TRAVELER NAMES ----------------
    if TRIP_MEMORY["step"] == "ask_names":
        TRIP_MEMORY["traveler_names"].append(user_input)

        if len(TRIP_MEMORY["traveler_names"]) < TRIP_MEMORY["adults"]:
            return jsonify({"reply": f"Enter name of traveler {len(TRIP_MEMORY['traveler_names']) + 1}"})

        TRIP_MEMORY["user_name"] = TRIP_MEMORY["traveler_names"][0]
        TRIP_MEMORY["step"] = "ask_age"
        return jsonify({"reply": "Enter primary passenger age"})

    # ---------------- ASK AGE ----------------
    if TRIP_MEMORY["step"] == "ask_age":
        TRIP_MEMORY["user_age"] = user_input

        TRIP_MEMORY["flights"] = search_flight.invoke({
            "origin": TRIP_MEMORY["source"],
            "destination": TRIP_MEMORY["destination"],
            "date": TRIP_MEMORY["start_date"]
        })

        # Update step in memory and return structured flight data
        TRIP_MEMORY["step"] = 2
        return jsonify({
            "reply": "Available Flights:",
            "flights": TRIP_MEMORY["flights"],
            "step": 2
        })

    # ---------------- STEP 2 ----------------
    if TRIP_MEMORY["step"] == 2:
        try:
            choice = int(user_input)
            if 1 <= choice <= len(TRIP_MEMORY["flights"]):
                TRIP_MEMORY["selected_flight"] = TRIP_MEMORY["flights"][choice - 1]
                TRIP_MEMORY["step"] = 3  # Move to hotel selection
                
                # Search for hotels
                TRIP_MEMORY["hotels"] = search_hotel.invoke({
                    "city": TRIP_MEMORY["destination"],
                    "check_in": TRIP_MEMORY["start_date"],
                    "check_out": TRIP_MEMORY["end_date"]
                })
                
                # Format hotel options
                hotel_options = [
                    f"{i+1}. {h['hotel_name']} - ₹{h['price']}/night"
                    for i, h in enumerate(TRIP_MEMORY["hotels"])
                ]
                
                return jsonify({
                    "reply": f"✅ Selected flight: {TRIP_MEMORY['selected_flight']['airline']} {TRIP_MEMORY['selected_flight']['flight_number']}\n\n"
                            f"Available Hotels:\n" + "\n".join(hotel_options) +
                            "\n\nPlease select a hotel by number:",
                    "step": 3
                })
            else:
                return jsonify({
                    "reply": "❌ Invalid selection. Please choose a valid flight number from the list.",
                    "flights": TRIP_MEMORY["flights"],
                    "step": 2
                })
        except ValueError:
            return jsonify({
                "reply": "❌ Please enter a valid number.",
                "flights": TRIP_MEMORY["flights"],
                "step": 2
            })

    # ---------------- STEP 3 ----------------
    if TRIP_MEMORY["step"] == 3:
        try:
            choice = int(user_input)
            if 1 <= choice <= len(TRIP_MEMORY["hotels"]):
                TRIP_MEMORY["selected_hotel"] = TRIP_MEMORY["hotels"][choice - 1]
                TRIP_MEMORY["step"] = 4  # Move to itinerary generation
                
                # Generate itinerary
                TRIP_MEMORY["itinerary"] = generate_trip_plan.invoke({
                    "start_date": TRIP_MEMORY["start_date"],
                    "end_date": TRIP_MEMORY["end_date"],
                    "source": TRIP_MEMORY["source"],
                    "destination": TRIP_MEMORY["destination"]
                })

                cost = calculate_total_cost()
                
                # Format the response with structured data
                return jsonify({
                    "reply": f"✅ Selected Hotel: {TRIP_MEMORY['selected_hotel']['hotel_name']}\n\n"
                            f"✅ ITINERARY GENERATED\n\n{TRIP_MEMORY['itinerary']}\n\n"
                            f"-------------------------\nTotal Cost: ₹{cost['total_cost']}\n\n"
                            f"Do you want to confirm and proceed to payment? (yes/no)",
                    "itinerary": TRIP_MEMORY["itinerary"],
                    "total_cost": cost['total_cost']
                })
            else:
                # Re-display hotel options if selection is invalid
                hotel_options = [
                    f"{i+1}. {h['hotel_name']} - ₹{h['price']}/night"
                    for i, h in enumerate(TRIP_MEMORY["hotels"])
                ]
                return jsonify({
                    "reply": "❌ Invalid selection. Please choose a valid hotel number:\n\n" + 
                            "\n".join(hotel_options),
                    "step": 3
                })
        except ValueError:
            # Re-display hotel options if input is not a number
            hotel_options = [
                f"{i+1}. {h['hotel_name']} - ₹{h['price']}/night"
                for i, h in enumerate(TRIP_MEMORY["hotels"])
            ]
            return jsonify({
                "reply": "❌ Please enter a valid number. Available hotels:\n\n" + 
                        "\n".join(hotel_options),
                "step": 3
            })


    # ---------------- STEP 4 PAYMENT ----------------
    if TRIP_MEMORY["step"] == 4:

        if user_input.lower() not in ["yes", "y"]:
            return jsonify({"reply": "Trip cancelled."})

        TRIP_MEMORY["payment_status"] = "PAID"
        trx = "TXN-" + str(uuid.uuid4())[:8]

        pdf_file = generate_booking_pdf()

        TRIP_MEMORY["step"] = "done"

        return jsonify({
            "reply": f"✅ Payment Successful\nTransaction ID: {trx}\n\nDownload your ticket:",
            "download": "/download"
        })


@app.route("/download")
def download():
    return send_file("trip_booking.pdf", as_attachment=True)

if __name__ == "__main__":
    app.run()
