import platform, os
import asyncio

from sources.utility import pretty_print, animate_thinking
from sources.agents.agent import Agent, executorResult
from sources.tools.C_Interpreter import CInterpreter
from sources.tools.GoInterpreter import GoInterpreter
from sources.tools.PyInterpreter import PyInterpreter
from sources.tools.BashInterpreter import BashInterpreter
from sources.tools.WindowsInterpreter import WindowsInterpreter
from sources.tools.JavaInterpreter import JavaInterpreter
from sources.tools.JavaScriptInterpreter import JavaScriptInterpreter
from sources.tools.HTMLInterpreter import HTMLInterpreter
from sources.tools.fileFinder import FileFinder
from sources.logger import Logger
from sources.memory import Memory
from sources.utility import get_os_type, is_running_in_docker

class CoderAgent(Agent):
    """
    The code agent is an agent that can write and execute code.
    """
    def __init__(self, name, prompt_path, provider, verbose=False):
        super().__init__(name, prompt_path, provider, verbose, None)
        
        # Use WindowsInterpreter on Windows when not in Docker
        if get_os_type() == 'windows' and not is_running_in_docker():
            bash_interpreter = WindowsInterpreter()
        else:
            bash_interpreter = BashInterpreter()
            
        self.tools = {
            "bash": bash_interpreter,
            "python": PyInterpreter(),
            "c": CInterpreter(),
            "go": GoInterpreter(),
            "java": JavaInterpreter(),
            "javascript": JavaScriptInterpreter(),
            "file_finder": FileFinder(),
            "html": HTMLInterpreter()
        }
        self.work_dir = self.tools["file_finder"].get_work_dir()
        self.role = "code"
        self.type = "code_agent"
        self.logger = Logger("code_agent.log")
        self.memory = Memory(self.load_prompt(prompt_path),
                        recover_last_session=False, # session recovery in handled by the interaction class
                        memory_compression=False,
                        model_provider=provider.get_model_name())
    
    def add_sys_info_prompt(self, prompt):
        """Add system information to the prompt."""
        info = f"System Info:\n" \
               f"OS: {platform.system()} {platform.release()}\n" \
               f"Python Version: {platform.python_version()}\n" \
               f"\nYou must save file at root directory: {self.work_dir}"
        return f"{prompt}\n\n{info}"

    async def process(self, prompt, speech_module) -> str:
        answer = ""
        attempt = 0
        max_attempts = 5
        prompt = self.add_sys_info_prompt(prompt)
        self.memory.push('user', prompt)
        clarify_trigger = "REQUEST_CLARIFICATION"

        while attempt < max_attempts and not self.stop:
            print("Stopped?", self.stop)
            animate_thinking("Thinking...", color="status")
            await self.wait_message(speech_module)
            answer, reasoning = await self.llm_request()
            self.last_reasoning = reasoning
            if clarify_trigger in answer:
                self.last_answer = answer
                await asyncio.sleep(0)
                return answer, reasoning
            if not "```" in answer:
                self.logger.warning("No code blocks found in LLM response!")
                self.memory.push('assistant', answer)
                self.memory.push('user', "Please provide the code in executable blocks using ``` tags as shown in the examples. For example:\n```python\nprint('Hello')\n```")
                attempt += 1
                continue  # Retry instead of breaking
            self.show_answer()
            animate_thinking("Executing code...", color="status")
            self.status_message = "Executing code..."
            self.logger.info(f"Attempt {attempt + 1}:\n{answer}")
            exec_success, feedback = self.execute_modules(answer)
            self.logger.info(f"Execution result: {exec_success}")
            answer = self.remove_blocks(answer)
            self.last_answer = answer
            await asyncio.sleep(0)
            if exec_success and self.get_last_tool_type() != "bash":
                break
            pretty_print(f"Execution failure:\n{feedback}", color="failure")
            pretty_print("Correcting code...", color="status")
            self.status_message = "Correcting code..."
            attempt += 1
        self.status_message = "Ready"
        if attempt == max_attempts:
            return "I'm sorry, I couldn't find a solution to your problem. How would you like me to proceed ?", reasoning
        self.last_answer = answer
        return answer, reasoning

if __name__ == "__main__":
    pass