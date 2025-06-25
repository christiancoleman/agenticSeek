import json
from typing import List, Tuple, Type, Dict
from sources.utility import pretty_print, animate_thinking
from sources.agents.agent import Agent
from sources.agents.code_agent import CoderAgent
from sources.agents.file_agent import FileAgent
from sources.agents.browser_agent import BrowserAgent
from sources.agents.casual_agent import CasualAgent
from sources.text_to_speech import Speech
from sources.tools.tools import Tools
from sources.logger import Logger
from sources.memory import Memory
from sources.conversation_logger import get_conversation_logger

class PlannerAgent(Agent):
    def __init__(self, name, prompt_path, provider, verbose=False, browser=None):
        """
        The planner agent is a special agent that divides and conquers the task.
        """
        super().__init__(name, prompt_path, provider, verbose, None)
        self.tools = {
            "json": Tools()
        }
        self.tools['json'].tag = "json"
        self.browser = browser
        self.agents = {
            "coder": CoderAgent("Coder", "prompts/base/coder_agent.txt", provider, verbose=False),
            "file": FileAgent("File", "prompts/base/file_agent.txt", provider, verbose=False),
            "web": BrowserAgent("Web", "prompts/base/browser_agent.txt", provider, verbose=False, browser=browser),
            "casual": CasualAgent("Casual", "prompts/base/casual_agent.txt", provider, verbose=False)
        }
        self.role = "planification"
        self.type = "planner_agent"
        self.tasks = []  # Initialize tasks list
        self.memory = Memory(self.load_prompt(prompt_path),
                                recover_last_session=False, # session recovery in handled by the interaction class
                                memory_compression=False,
                                model_provider=provider.get_model_name())
        self.logger = Logger("planner_agent.log")
    
    def get_task_names(self, text: str) -> List[str]:
        """
        Extracts task names from the given text.
        This method processes a multi-line string, where each line may represent a task name.
        containing '##' or starting with a digit. The valid task names are collected and returned.
        Args:
            text (str): A string containing potential task titles (eg: Task 1: I will...).
        Returns:
            List[str]: A list of extracted task names that meet the specified criteria.
        """
        tasks_names = []
        lines = text.strip().split('\n')
        for line in lines:
            if line is None:
                continue
            line = line.strip()
            if len(line) == 0:
                continue
            if '##' in line or line[0].isdigit():
                tasks_names.append(line)
                continue
        self.logger.info(f"Found {len(tasks_names)} tasks names.")
        return tasks_names

    def parse_agent_tasks(self, text: str) -> List[Tuple[str, str]]:
        """
        Parses agent tasks from the given LLM text.
        This method extracts task information from a JSON. It identifies task names and their details.
        Args:
            text (str): The input text containing task information in a JSON-like format.
        Returns:
            List[Tuple[str, str]]: A list of tuples containing task names and their details.
        """
        tasks = []
        tasks_names = self.get_task_names(text)

        blocks, _ = self.tools["json"].load_exec_block(text)
        if blocks is None or len(blocks) == 0:
            self.logger.warning("No JSON blocks found in LLM response")
            return []
        for block in blocks:
            try:
                line_json = json.loads(block)
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse JSON block: {e}")
                continue
                
            # Handle both {"plan": [...]} and direct task format
            if 'plan' in line_json:
                # Standard format with plan array
                for task in line_json['plan']:
                    # Ensure task is a dictionary
                    if not isinstance(task, dict):
                        self.logger.warning(f"Task is not a dictionary: {type(task)} - {task}")
                        continue
                    
                    # Check if agent key exists
                    if 'agent' not in task:
                        self.logger.warning(f"Task missing 'agent' field: {task}")
                        continue
                        
                    if task['agent'].lower() not in [ag_name.lower() for ag_name in self.agents.keys()]:
                        self.logger.warning(f"Agent {task['agent']} does not exist.")
                        pretty_print(f"Agent {task['agent']} does not exist.", color="warning")
                        # Log the error to conversation logger
                        conv_logger = get_conversation_logger()
                        conv_logger.log_planner_error(f"Agent {task['agent']} does not exist", text)
                        return []
                    try:
                        agent = {
                            'agent': task['agent'],
                            'id': task.get('id', str(len(tasks) + 1)),  # Default ID if missing
                            'task': task.get('task', 'No task description')  # Default description
                        }
                    except Exception as e:
                        self.logger.warning(f"Error creating agent dict: {e}")
                        return []
                    self.logger.info(f"Created agent {task['agent']} with task: {task.get('task', 'No description')}")
                    if 'need' in task:
                        self.logger.info(f"Agent {task['agent']} was given info:\n {task['need']}")
                        agent['need'] = task['need']
                    tasks.append(agent)
            elif 'agent' in line_json and 'task' in line_json:
                # Handle single task format (when LLM returns just one task)
                self.logger.info(f"Detected single task format, converting to plan")
                task = line_json
                if task['agent'].lower() not in [ag_name.lower() for ag_name in self.agents.keys()]:
                    self.logger.warning(f"Agent {task['agent']} does not exist.")
                    pretty_print(f"Agent {task['agent']} does not exist.", color="warning")
                    # Log the error to conversation logger
                    conv_logger = get_conversation_logger()
                    conv_logger.log_planner_error(f"Agent {task['agent']} does not exist", text)
                    return []
                try:
                    agent = {
                        'agent': task['agent'],
                        'id': task.get('id', '1'),
                        'task': task.get('task', 'No task description')
                    }
                    if 'need' in task:
                        agent['need'] = task['need']
                    tasks.append(agent)
                except Exception as e:
                    self.logger.warning(f"Error creating agent dict from single task: {e}")
                    return []
        if len(tasks_names) != len(tasks):
            names = [task['task'] for task in tasks]
            return list(map(list, zip(names, tasks)))
        return list(map(list, zip(tasks_names, tasks)))
    
    def make_prompt(self, task: str, agent_infos_dict: dict) -> str:
        """
        Generates a prompt for the agent based on the task and previous agents work information.
        Args:
            task (str): The task to be performed.
            agent_infos_dict (dict): A dictionary containing information from other agents.
        Returns:
            str: The formatted prompt for the agent.
        """
        infos = ""
        if agent_infos_dict is None or len(agent_infos_dict) == 0:
            infos = "No needed informations."
        else:
            for agent_id, info in agent_infos_dict.items():
                infos += f"\t- According to agent {agent_id}:\n{info}\n\n"
        prompt = f"""
        You are given informations from your AI friends work:
        {infos}
        Your task is:
        {task}
        """
        self.logger.info(f"Prompt for agent:\n{prompt}")
        return prompt
    
    def show_plan(self, agents_tasks: List[dict], answer: str) -> None:
        """
        Displays the plan made by the agent.
        Args:
            agents_tasks (dict): The tasks assigned to each agent.
            answer (str): The answer from the LLM.
        """
        if agents_tasks == []:
            pretty_print(answer, color="warning")
            pretty_print("Failed to make a plan. This can happen with (too) small LLM. Clarify your request and insist on it making a plan within ```json.", color="failure")
            return
        pretty_print("\n▂▘ P L A N ▝▂", color="status")
        for task_name, task in agents_tasks:
            pretty_print(f"{task['agent']} -> {task['task']}", color="info")
        pretty_print("▔▗ E N D ▖▔", color="status")

    async def make_plan(self, prompt: str, is_update: bool = False) -> str:
        """
        Asks the LLM to make a plan.
        Args:
            prompt (str): The prompt to be sent to the LLM.
            is_update (bool): Whether this is a plan update or initial plan.
        Returns:
            str: The plan made by the LLM.
        """
        ok = False
        answer = None
        while not ok:
            animate_thinking("Thinking...", color="status")
            self.memory.push('user', prompt)
            answer, reasoning = await self.llm_request()
            if "NO_UPDATE" in answer:
                return []
            self.logger.info(f"LLM response for plan update:\n{answer}")
            
            # Extract reasoning (everything before ```json)
            json_start = answer.find("```json")
            if json_start > 0:
                plan_reasoning = answer[:json_start].strip()
                if plan_reasoning and not is_update:  # Only log initial plan reasoning
                    conv_logger = get_conversation_logger()
                    conv_logger.log_planner_reasoning(plan_reasoning)
            
            agents_tasks = self.parse_agent_tasks(answer)
            if agents_tasks == []:
                self.show_plan(agents_tasks, answer)
                prompt = f"Failed to parse the tasks. Please write down your task followed by a json plan within ```json. Do not ask for clarification.\n"
                pretty_print("Failed to make plan. Retrying...", color="warning")
                # Log the failed plan attempt
                conv_logger = get_conversation_logger()
                conv_logger.log_planner_error("Failed to parse tasks from LLM response", answer)
                continue
            self.show_plan(agents_tasks, answer)
            ok = True
        self.logger.info(f"Plan made:\n{answer}")
        # Log the plan to conversation logger
        conv_logger = get_conversation_logger()
        conv_logger.log_planner_plan(agents_tasks, is_update=is_update)
        return self.parse_agent_tasks(answer)
    
    async def update_plan(self, goal: str, agents_tasks: List[dict], agents_work_result: dict, id: str, success: bool) -> dict:
        """
        Updates the plan with the results of the agents work.
        Args:
            goal (str): The goal to be achieved.
            agents_tasks (list): The tasks assigned to each agent.
            agents_work_result (dict): The results of the agents work.
        Returns:
            dict: The updated plan.
        """
        self.status_message = "Updating plan..."
        last_agent_work = agents_work_result[id]
        tool_success_str = "success" if success else "failure"
        pretty_print(f"Agent {id} work {tool_success_str}.", color="success" if success else "failure")
        try:
            id_int = int(id)
        except Exception as e:
            return agents_tasks
        if id_int == len(agents_tasks):
            next_task = "No task follow, this was the last step. If it failed add a task to recover."
        else:
            next_task = f"Next task is: {agents_tasks[int(id)][0]}."
        #if success:
        #    return agents_tasks # we only update the plan if last task failed, for now
        update_prompt = f"""
        Your goal is : {goal}
        You previously made a plan, agents are currently working on it.
        The last agent working on task: {id}, did the following work:
        {last_agent_work}
        Agent {id} work was a {tool_success_str} according to system interpreter.
        {next_task}
        Is the work done for task {id} leading to success or failure ? Did an agent fail with a task?
        If agent work was good: answer "NO_UPDATE"
        If agent work is leading to failure: update the plan with the EXACT format:
        ```json
        {{
          "plan": [
            {{
              "agent": "agent_name",
              "id": "task_id",
              "need": ["dependency_ids"],
              "task": "task_description"
            }}
          ]
        }}
        ```
        You need to rewrite the whole plan, but only change the tasks after task {id}.
        Make the plan the same length as the original one or with only one additional step.
        Do not change past tasks. Change next tasks.
        """
        pretty_print("Updating plan...", color="status")
        plan = await self.make_plan(update_prompt, is_update=True)
        if plan == []:
            pretty_print("No plan update required.", color="info")
            conv_logger = get_conversation_logger()
            conv_logger.log_no_update_decision()
            return agents_tasks
        self.logger.info(f"Plan updated:\n{plan}")
        return plan
    
    async def start_agent_process(self, task: dict, required_infos: dict | None) -> str:
        """
        Starts the agent process for a given task.
        Args:
            task (dict): The task to be performed.
            required_infos (dict | None): The required information for the task.
        Returns:
            str: The result of the agent process.
        """
        # Validate task is a dictionary with required fields
        if not isinstance(task, dict):
            self.logger.error(f"Task is not a dictionary: {type(task)} - {task}")
            return ("Error: Invalid task format", False)
        
        required_fields = ['task', 'agent', 'id']
        for field in required_fields:
            if field not in task:
                self.logger.error(f"Task missing required field '{field}': {task}")
                return (f"Error: Task missing {field} field", False)
        
        self.status_message = f"Starting task {task['task']}..."
        agent_prompt = self.make_prompt(task['task'], required_infos)
        
        # Log agent conversation start
        conv_logger = get_conversation_logger()
        conv_logger.start_agent_conversation(task['agent'], task['id'], agent_prompt)
        
        pretty_print(f"Agent {task['agent']} started working...", color="status")
        self.logger.info(f"Agent {task['agent']} started working on {task['task']}.")
        answer, reasoning = await self.agents[task['agent'].lower()].process(agent_prompt, None)
        self.last_answer = answer
        self.last_reasoning = reasoning
        
        self.blocks_result = self.agents[task['agent'].lower()].blocks_result
        agent_answer = self.agents[task['agent'].lower()].raw_answer_blocks(answer)
        
        # Log agent response with expanded blocks
        conv_logger.log_agent_response(task['agent'], agent_answer, reasoning)
        success = self.agents[task['agent'].lower()].get_success
        
        # Log execution result
        conv_logger.log_execution_result(success, agent_answer if not success else None)
        
        self.agents[task['agent'].lower()].show_answer()
        pretty_print(f"Agent {task['agent']} completed task.", color="status")
        self.logger.info(f"Agent {task['agent']} finished working on {task['task']}. Success: {success}")
        agent_answer += "\nAgent succeeded with task." if success else "\nAgent failed with task (Error detected)."
        
        # End agent conversation
        conv_logger.end_agent_conversation(task['agent'])
        
        return agent_answer, success
    
    def get_work_result_agent(self, task_needs, agents_work_result):
        res = {k: agents_work_result[k] for k in task_needs if k in agents_work_result}
        self.logger.info(f"Next agent needs: {task_needs}.\n Match previous agent result: {res}")
        return res

    async def process(self, goal: str, speech_module: Speech) -> Tuple[str, str]:
        """
        Process the goal by dividing it into tasks and assigning them to agents.
        Args:
            goal (str): The goal to be achieved (user prompt).
            speech_module (Speech): The speech module for text-to-speech.
        Returns:
            Tuple[str, str]: The result of the agent process and empty reasoning string.
        """
        agents_tasks = []
        required_infos = None
        agents_work_result = dict()

        self.status_message = "Making a plan..."
        agents_tasks = await self.make_plan(goal)

        if agents_tasks == []:
            return "Failed to parse the tasks.", ""
        i = 0
        steps = len(agents_tasks)
        while i < steps and not self.stop:
            task_name, task = agents_tasks[i][0], agents_tasks[i][1]
            self.status_message = "Starting agents..."
            pretty_print(f"I will {task_name}.", color="info")
            self.last_answer = f"I will {task_name.lower()}."
            pretty_print(f"Assigned agent {task['agent']} to {task_name}", color="info")
            if speech_module: speech_module.speak(f"I will {task_name}. I assigned the {task['agent']} agent to the task.")

            if agents_work_result is not None:
                required_infos = self.get_work_result_agent(task['need'], agents_work_result)
            try:
                answer, success = await self.start_agent_process(task, required_infos)
            except Exception as e:
                raise e
            if self.stop:
                pretty_print(f"Requested stop.", color="failure")
            agents_work_result[task['id']] = answer
            agents_tasks = await self.update_plan(goal, agents_tasks, agents_work_result, task['id'], success)
            steps = len(agents_tasks)
            i += 1

        # Log final result
        conv_logger = get_conversation_logger()
        conv_logger.log_final_result(answer)
        
        return answer, ""
