# Week 1 - Project 1: Rule-Based AI Chatbot 🤖

## DecodeLabs Industrial Training Kit (Batch 2026)

### Goal
Create a simple rule-based chatbot that responds to predefined user inputs
using control flow (if-else / dictionary lookups) inside a continuous loop.

### Key Requirements (from spec)
- [x] **Input Loop** – Continuous `while True` cycle
- [x] **Sanitization** – Handles case & whitespace (`.lower().strip()`)
- [x] **Knowledge Base** – Dictionary with 8 intents (5+ required)
- [x] **Fallback** – Default response for unrecognized input
- [x] **Exit Strategy** – Clean break on `bye` / `exit` / `quit`

### How to Run
```bash
python chatbot.py
```

### Example Session
DecodeBot - DecodeLabs Rule-Based AI Chatbot (Project 1)

Type 'help' for sample commands, or 'bye'/'exit' to quit.

You: hello

DecodeBot: Hi there! How can I help you today?

You: what can you do

DecodeBot: I can respond to greetings and simple questions using predefined rules. Type 'help' for a list.

You: bye

DecodeBot: Goodbye! Have a great day.

### Key Skills Practiced
- Control flow & decision-making logic
- Dictionary-based "intent matching" (`responses.get()`)
- Building a deterministic, traceable "white box" system — the
  foundation before adding any generative/LLM layer

### Possible Extensions (from the Conclusion section of the kit)
- Expand the bot's vocabulary with more intents
- Add nested conditions for smarter, multi-turn responses
- Give the bot a unique personality / theme
