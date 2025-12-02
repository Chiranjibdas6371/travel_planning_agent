# 🧠 AI Travel Planning Agent

A step-based, conversational AI Travel Planning Agent built with Python. The agent understands natural language trip requests, shows flights and hotels, generates a multi-day itinerary, calculates total trip cost, performs a dummy payment, and finally generates a booking file with all details.

---

## 🚀 Features

* Natural language trip creation
* Step-based workflow (no skipping steps)
* Flight search using API
* Hotel search using API
* Multi-day AI-generated itinerary
* User details collection (name, age, number of adults)
* Automatic trip cost calculation
* Dummy payment simulation
* Automatic ticket & itinerary file generation
* Fully terminal-based conversational flow

---

## 🧩 Workflow

1. User creates trip using natural language
2. System shows available flights
3. User selects a flight
4. System shows available hotels
5. User selects a hotel
6. AI generates multi-day itinerary
7. Total trip cost is calculated
8. User confirms and makes dummy payment
9. Booking file is generated

---

## 🛠️ Tech Stack

* Python 3.10+
* LangChain
* Google Gemini API
* Amadeus API (Flights & Hotels)
* Regex-based NLP Parsing

---

## 📦 Installation

```bash
pip install langchain langchain-google-genai python-dotenv requests
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root and add:

```env
GEMINI_API_KEY=your_google_gemini_key
AMADEUS_API_KEY=your_amadeus_api_key
AMADEUS_API_SECRET=your_amadeus_api_secret
```

---

## ▶️ How to Run

```bash
python travel_agent.py
```

---

## 💬 Example Input

```
Create a trip from Delhi to Mumbai from 2025-12-10 to 2025-12-15
```

---

## 🧾 Output

* Flight options
* Hotel options
* Multi-day itinerary
* Total trip cost
* Booking confirmation file: `trip_booking.txt`

---

## 🗂️ Generated Files

* `trip_booking.txt` – Contains:

  * Passenger details
  * Flight details
  * Hotel booking
  * Itinerary
  * Total amount paid

---

## ✅ Supported Commands

* Natural language trip creation
* Numeric selection for flight & hotel
* yes / no for payment confirmation
* exit / quit to end session

---

## ⚠️ Notes

* This project uses dummy payment and simulated booking.
* No real transactions are performed.
* Booking file is generated locally.

---

## 🔮 Future Enhancements

* PDF ticket generation
* Web-based UI with Flask
* Real payment gateway integration
* User authentication
* Trip history dashboard

---

## 👨‍💻 Author

Developed by Chiranjib

---

## 📜 License

This project is for educational and research purposes only.
