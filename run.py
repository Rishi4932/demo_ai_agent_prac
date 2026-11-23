
from pathlib import Path

from core.agent_controller import AgentController


PROJECT_ROOT = Path(__file__).resolve().parent


def create_required_directories():
    directories = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "data" / "screenshots",
        PROJECT_ROOT / "data" / "logs",
        PROJECT_ROOT / "data" / "database",
        PROJECT_ROOT / "data" / "voice",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def print_agent_result(result):
    print()
    print("=" * 60)
    print(" AGENT RESULT")
    print("=" * 60)
    print()
    print(result)


def run_typed_task(controller):
    print()
    print("=" * 60)
    print(" TYPED TASK")
    print("=" * 60)
    print()

    task = input("Enter your task: ").strip()

    if not task:
        print("No task entered.")
        return

    print()
    print("TASK:")
    print(task)
    print()

    try:
        controller.submit_task(task)

        print("Starting agent...")
        print()

        result = controller.run()
        print_agent_result(result)

    except Exception as exc:
        print()
        print("Agent execution failed:")
        print(exc)


def run_voice_task(controller):
    # Import voice components only when the user selects voice input.
    # This keeps typed-task mode independent of Whisper/Numba dependencies.
    try:
        from voice.voice_commands import VoiceCommandProcessor
    except Exception as exc:
        print()
        print("Voice recognition is currently unavailable.")
        print("You can still use option 1 to run typed tasks.")
        print(f"Voice dependency error: {exc}")
        return

    print()
    print("=" * 60)
    print(" VOICE TASK")
    print("=" * 60)
    print()

    try:
        voice_processor = VoiceCommandProcessor(
            whisper_model="base",
            recording_duration=10.0,
            max_attempts=3,
        )

        task = voice_processor.listen(
            filename="voice_command.wav"
        )

        if not task:
            print()
            print("No voice command was received.")
            print("Returning to the main menu.")
            return

        print()
        print("=" * 60)
        print(" VOICE TASK RECEIVED")
        print("=" * 60)
        print()
        print(f"Task: {task}")
        print()

        controller.submit_task(task)

        print("Starting agent...")
        print()

        result = controller.run()
        print_agent_result(result)

    except Exception as exc:
        print()
        print("Voice agent execution failed:")
        print(exc)


def main():
    create_required_directories()

    while True:
        print()
        print("=" * 60)
        print(" COMPUTER USE AI AGENT")
        print("=" * 60)
        print()
        print(f"Project directory: {PROJECT_ROOT}")
        print()
        print("Select how you want to provide the task:")
        print()
        print("1. Type a task")
        print("2. Speak a task")
        print("3. Exit")
        print()

        choice = input("Enter your choice: ").strip()

        if choice == "3":
            print()
            print("Exiting...")
            break

        controller = AgentController()

        if choice == "1":
            run_typed_task(controller)

        elif choice == "2":
            run_voice_task(controller)

        else:
            print()
            print("Invalid choice.")
            print("Please select 1, 2, or 3.")


if __name__ == "__main__":
    main()
