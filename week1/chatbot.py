
"""
DecodeLabs Internship - Project 1
Rule-Based AI Chatbot

Goal:
A simple rule-based chatbot that responds to predefined user inputs
using a continuous loop, dictionary-based "intent matching" (the
.get() method), and a clean exit strategy.

Key Skills demonstrated:
- Control flow (while loop, if-else)
- Input sanitization (lower-casing & stripping whitespace)
- Dictionary-based knowledge base (O(1) lookups)
- Fallback handling for unknown inputs
"""


def get_response(clean_input: str, responses: dict) -> str:
    """
    Look up a response for the given (already sanitized) input.
    Falls back to a default message if no match is found.
    This is the 'atomic lookup + fallback' pattern from the spec.
    """
    return responses.get(clean_input, "I do not understand that. Type 'help' to see what I can respond to.")


def build_knowledge_base() -> dict:
    """
    The 'Logic Skeleton' - a dictionary acting as the chatbot's
    knowledge base. Keys are expected user intents (already lower-cased),
    values are the bot's replies.
    """
    return {
        "hello": "Hi there! How can I help you today?",
        "hi": "Hello! Welcome to the DecodeLabs Chatbot.",
        "how are you": "I'm just a program, but I'm running perfectly fine!",
        "what is your name": "I am DecodeBot, your friendly rule-based assistant.",
        "what can you do": "I can respond to greetings and simple questions using predefined rules. Type 'help' for a list.",
        "help": "Try saying: hello, hi, how are you, what is your name, what can you do, thank you, bye/exit",
        "thank you": "You're welcome! Happy to help.",
        "thanks": "You're welcome!",
    }


def main():
    responses = build_knowledge_base()
    exit_commands = ("bye", "exit", "quit")

    print("=" * 55)
    print(" DecodeBot - DecodeLabs Rule-Based AI Chatbot (Project 1)")
    print(" Type 'help' for sample commands, or 'bye'/'exit' to quit.")
    print("=" * 55)

    # PHASE: The Heartbeat - Infinite Loop until the Kill Command
    while True:
        raw_input_text = input("You: ")

        # PHASE 1: Input & Sanitization
        clean_input = raw_input_text.lower().strip()

        # EXIT STRATEGY: Clean break command
        if clean_input in exit_commands:
            print("DecodeBot: Goodbye! Have a great day.")
            break

        # Skip empty input gracefully
        if clean_input == "":
            print("DecodeBot: Please type something, or 'bye' to exit.")
            continue

        # PROCESS: Intent Matching via dictionary .get()
        reply = get_response(clean_input, responses)

        # OUTPUT: Response Generation
        print("DecodeBot:", reply)


if __name__ == "__main__":
    main()
