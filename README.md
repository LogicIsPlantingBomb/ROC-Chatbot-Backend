# ROC AI Chatbot Backend

## Overview

This microservice handles AI chatbot interactions for the ROC vehicle booking app. The chatbot assists users by suggesting travel destinations, checking weather conditions, and evaluating travel feasibility based on vehicle type and location.

Built using Python and FastAPI, this service integrates with the main ROC app via REST APIs and WebSockets for real-time responses.

---

## Features

* Accepts user queries about travel destinations.
* Provides weather updates for planned routes.
* Suggests optimal destinations based on user preferences.
* Real-time communication with users through WebSocket notifications.

---

## Tech Stack

* **Language:** Python 3.x
* **Framework:** FastAPI
* **WebSocket:** FastAPI WebSocket
* **AI Integration:** OpenAI GPT-based APIs or custom ML models
* **Deployment:** Compatible with Docker for microservices architecture

---

## Project Structure

```
ChatBot/
├── app/                 # FastAPI application modules
│   ├── api/             # API route definitions
│   ├── models/          # Pydantic models
│   ├── services/        # AI service logic, e.g., OpenAI interaction
│   └── utils/           # Helper functions
├── main.py              # FastAPI app entry point
├── run.py               # Script to start the server
├── requirements.txt     # Python dependencies
├── myenv/               # Virtual environment
└── README.md
```

---

## Installation

1. Clone the repository:

```bash
git clone <repo-url>
cd ChatBot
```

2. Create and activate virtual environment (if not already):

```bash
python3 -m venv myenv
source myenv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the FastAPI server:

```bash
uvicorn main:app --reload
```

The chatbot backend will now run at `http://localhost:8000`.

---

## WebSocket Example

### main.py snippet

```python
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        response = handle_chat_query(data)  # Custom AI logic
        await websocket.send_text(response)
```

---

## Contributing

* Maintain modularity between AI logic, API routes, and WebSocket communication.
* Write clean and documented code for AI interaction.
* Ensure integration with main ROC app via REST or WebSocket is consistent.

---

## License

MIT License
