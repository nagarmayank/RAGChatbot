from utils.helper_methods import pretty_print_messages
from agents.supervisor_agent import supervisor_agent

print("Supervisor Agent is ready to use.")
while True:
    user_message = input("Enter your message (or type 'exit' to quit): ")
    if user_message.lower() == 'exit':
        print("Exiting the supervisor agent.")
        break

    message = [{"role":"user", "content": user_message}]

    for chunk in supervisor_agent().stream({"messages": message}):
        pretty_print_messages(chunk)