import os
import subprocess
from chatbots import *

available_commands = [
    "CMD: remove_directory <directory name> # remove a directory",
    "CMD: remove_file <file name> # remove a file",
    "CMD: make_directory <directory name> # make a new empty file with a given name",
    "CMD: make_file <file name> # make a new empty file with a given name",
    "CMD: move <source file or directory> <destination> # move a file or directory to a new location",
    "CMD: read_file <file name> # this one dumps the text of a file",
    # "CMD: retrieve <number of documents> <content> # this one uses a retrieval mechanism to find some number of documents that are most semantically similar to the text provided",
    "CMD: write <filename> <content> # write text to a file (this will overwrite any content already there, so if you want to edit a file that already exists, you need to generate the entire file with your proposed changes, sorry!). This will probably be your most used command, as you will be generating and then writing a lot of code.",
    "CMD: append <filename> <content> # append new text to a file",
    "CMD: <other relevant command> # if none of the above feel right, feel free to suggest any valid bash command that will do the job. Examples include package installation, compilation, make, execution, or other things. Just make sure that the command is valid bash",
]

command_list = "\n".join(available_commands)

def terminal_info(completed_process):
        result = ""#f"Agent % {completed_process.args}\n"
        if completed_process.stdout:
            result += f"{str(completed_process.stdout, encoding='utf-8')}\n"
        if completed_process.stderr:
            result += f"{str(completed_process.stderr, encoding='utf-8')}\n"
        return result

def directory_info(dir_name, prefix=''):
    if not os.path.exists(dir_name):
        return f"The directory {dir_name} does not exist.\n"
    
    dir_name = os.path.abspath(dir_name)
    result = f"{prefix}{os.path.basename(dir_name)}/\n"  # Start with the current directory name
    files_and_dirs = os.listdir(dir_name)
    files_and_dirs.sort()  # Sort the list for consistent order
    entries = [os.path.join(dir_name, fd) for fd in files_and_dirs]

    # Process directories and files separately
    dirs = [d for d in entries if os.path.isdir(d)]
    files = [f for f in entries if os.path.isfile(f)]

    # Recursively process each directory and collect results
    for d in dirs:
        result += directory_info(d, prefix + "│   ")
    
    # Append files to the result
    for f in files:
        result += f"{prefix}│   └── {os.path.basename(f)}\n"

    return result

def make_system_prompt(working_directory):
        return 

class AkashicArchitect():
    def __init__(self, chatbot, working_directory):
        self.working_directory = working_directory
        self.agent = chatbot
        self.agent.system_prompt = f"""You are an LLM software engineer.
        Here are all the possible commands you could execute,
        each one is followed by a comment explaining its use:\n
        {command_list}\n
        Reply in this format, and do not stray from this format:
        "CMD: <command to execute>"
        Do not stray from this format.
        Every response you give should be a single command from the list above.
        You are one level above {working_directory}.
        You cannot change directory, ever. Every path you give me must begin with {working_directory}.
        A human will verify any actions you take.
        Do not reply with any text besides the command you want to execute, in the format above,
        and please only reply with a single command, rather than multiple."""

    def retrieve_docs(self, n, content):
        return "No matching documents"

    def create_user_messages(self, objective, term_info):
        return f"""
    Result of your last action (if any): \n{term_info}\n
    Human input, given your last action: \n{objective}\n
    Current directory layout: \n{directory_info(self.working_directory)}\n
    Now, drawing from the list of commands given, please suggest the next action to take.
    Remember to stick to the format, and only reply with a single command.
    """

    def parse_response(self, response):
        # print("Raw response:", response)
        if not response.startswith(" CMD: "):
            return 'Invalid response', "format"
        response = response[6:]
        response = response.split()

        match response[0]:
            case "cd":
                return "Null", "cd"
            
            case "remove_directory":
                if not response[1].startswith(self.working_directory):
                    return "null", "path"
                return f"rm -r {response[1]}", "shell"

            case "remove_file":
                if not response[1].startswith(self.working_directory):
                    return "null", "path"
                return f"rm {response[1]}", "shell"
            
            case "make_directory":
                if not response[1].startswith(self.working_directory):
                    return "null", "path"
                return f"mkdir {response[1]}", "shell"

            case "make_file":
                if not response[1].startswith(self.working_directory):
                    return "null", "path"
                return f"touch {response[1]}", "shell"
            
            case "move":
                if not response[1].startswith(self.working_directory) or not response[2].startswith(self.working_directory):
                    return "null", "path"
                return f"mv {response[1]} {response[2]}", "shell"
            
            case "read_file":
                if not response[1].startswith(self.working_directory):
                    return "null", "path"
                return f"cat {response[1]}", "shell"
            
            case "retrieve":
                return " ".join(response), "custom"
            
            case "write":
                return " ".join(response), "custom"
            
            case "append":
                return " ".join(response), "custom"
            
            case _:
                return " ".join(response), "shell"

    def run_command(self, command, command_type):
        if command_type == "shell":
            return subprocess.run(command, shell=True, capture_output=True)
        else:
            command = command.split()
            match command[0]:
                case "retrieve":
                    return self.retrieve_docs(command[1], " ".join(command[2:]))
                case "write":
                    with open(command[1], 'w') as file:
                        file.write((" ".join(command[2:])[1:-1]).encode('utf-8').decode('unicode_escape'))
                    return f"Successfully wrote content to file {command[1]}"
                case "append":
                    with open(command[1], 'a') as file:
                        file.write((" ".join(command[2:])[1:-1]).encode('utf-8').decode('unicode_escape'))
                    return f"Successfully appended content to file {command[1]}"

    def loop(self):
        term_info = ""
        objective = input("User: ")

        usr_msg = self.create_user_messages(objective, term_info)
        print(usr_msg)

        while objective != "STOP":
            response = self.agent.chat_text(usr_msg)
            command, cmd_type = self.parse_response(response)
            if cmd_type == "shell" or cmd_type == "custom":
                print(f"Agent: {command}")

                allowance = input("\nIs this okay? [y/n]: ").lower()
                while allowance != 'y' and allowance != 'n':
                    allowance = input("\nIs this okay? [y/n]: ").lower()
                if allowance == 'y':
                    action = self.run_command(command, cmd_type)
                    if cmd_type == "shell":
                        term_info = terminal_info(action)
                    else:
                        term_info = action
                else:
                    term_info = f"No terminal output for this command, a human user denied it."

                objective = input("User: ")
                usr_msg = self.create_user_messages(objective, term_info=term_info)
            else:
                if cmd_type == "cd":
                                usr_msg = self.create_user_messages(f'You are not allowed to change directory. Alter your relative paths to work from one directory above {project_directory}', term_info="No terminal output for this command, a human user denied it.")
                elif cmd_type == "format":
                    usr_msg = self.create_user_messages('Your reply was not formatted correctly. All replies should look like this: "CMD: <command from the list I gave you>"', term_info="No terminal output for this command, a human user denied it.")
                elif cmd_type == "path":
                    usr_msg = self.create_user_messages(f'Your relative paths should all begin with {self.working_directory}, and then build from there.', term_info="No terminal output for this command, a human user denied it.")
                else:
                    usr_msg = self.create_user_messages('Your reply was not formatted correctly. All replies should look like this: "CMD: <command from the list I gave you>"', term_info="No terminal output for this command, a human user denied it.")

if __name__ == "__main__":
    project_directory = "project_directory"
    chatter = neural_chat_chatbot_Q4(context_length=2048)
    architect = AkashicArchitect(chatter, project_directory)
    architect.loop()

