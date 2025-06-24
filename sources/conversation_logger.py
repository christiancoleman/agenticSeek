"""
Conversation Logger Module
A standalone module for logging agent conversations sequentially.
Can be easily removed without affecting core functionality.
"""

import os
import json
import datetime
from typing import Optional, Dict, Any
from pathlib import Path


class ConversationLogger:
    """Logs agent conversations in a readable sequential format."""
    
    def __init__(self, enabled: bool = True, log_dir: str = ".conversation_logs"):
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self.current_session_id = None
        self.current_log_file = None
        self.conversation_stack = []
        
        if self.enabled:
            self.log_dir.mkdir(exist_ok=True)
            self._start_new_session()
    
    def _start_new_session(self):
        """Start a new conversation logging session."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session_id = f"conversation_{timestamp}"
        self.current_log_file = self.log_dir / f"{self.current_session_id}.md"
        
        # Write header
        with open(self.current_log_file, 'w', encoding='utf-8') as f:
            f.write(f"# Conversation Log - {timestamp}\n\n")
            f.write("This log shows the sequential flow of agent conversations.\n\n")
            f.write("---\n\n")
    
    def log_user_query(self, query: str):
        """Log the initial user query."""
        if not self.enabled:
            return
            
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            f.write(f"## 🧑 User Query\n")
            f.write(f"**Time**: {datetime.datetime.now().strftime('%H:%M:%S')}\n\n")
            f.write(f"```\n{query}\n```\n\n")
            f.write("---\n\n")
    
    def log_router_decision(self, agent_type: str, complexity: str = None):
        """Log the router's decision."""
        if not self.enabled:
            return
            
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            f.write(f"## 🚦 Router Decision\n")
            f.write(f"**Selected Agent**: {agent_type}\n")
            if complexity:
                f.write(f"**Complexity**: {complexity}\n")
            f.write("\n---\n\n")
    
    def log_planner_plan(self, plan: list, is_update: bool = False):
        """Log the planner's task breakdown."""
        if not self.enabled:
            return
            
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            if is_update:
                f.write(f"## 🔄 Updated Plan\n\n")
                f.write("*The planner has revised the task breakdown:*\n\n")
            else:
                f.write(f"## 📋 Planner's Task Breakdown\n\n")
                
            for i, task_item in enumerate(plan, 1):
                # Handle both list format [task_name, task_dict] and direct dict format
                if isinstance(task_item, list) and len(task_item) >= 2:
                    task_name, task = task_item[0], task_item[1]
                    f.write(f"### Task {i}: {task_name}\n")
                    f.write(f"**Agent**: {task.get('agent', 'Unknown')}\n")
                    f.write(f"**ID**: {task.get('id', 'N/A')}\n")
                    f.write(f"**Dependencies**: {task.get('need', [])}\n")
                    f.write(f"**Description**: {task.get('task', 'No description')}\n\n")
                elif isinstance(task_item, dict):
                    # Direct dict format
                    f.write(f"### Task {i}: {task_item.get('agent', 'Unknown')}\n")
                    f.write(f"**ID**: {task_item.get('id', 'N/A')}\n")
                    f.write(f"**Dependencies**: {task_item.get('need', [])}\n")
                    f.write(f"**Description**: {task_item.get('task', 'No description')}\n\n")
            f.write("---\n\n")
    
    def log_planner_error(self, error_msg: str, raw_response: str = None):
        """Log planner errors and retry attempts."""
        if not self.enabled:
            return
            
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            f.write(f"## ⚠️ Planner Error\n\n")
            f.write(f"**Error**: {error_msg}\n\n")
            if raw_response:
                f.write(f"<details>\n<summary>Raw LLM Response</summary>\n\n```\n{raw_response}\n```\n</details>\n\n")
            f.write("*Retrying...*\n\n")
            f.write("---\n\n")
    
    def log_planner_reasoning(self, reasoning: str):
        """Log the planner's reasoning before the task breakdown."""
        if not self.enabled:
            return
            
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            f.write(f"## 🤔 Planner's Reasoning\n\n")
            f.write(f"```\n{reasoning}\n```\n\n")
            f.write("---\n\n")
    
    def log_no_update_decision(self):
        """Log when planner decides no update is needed."""
        if not self.enabled:
            return
            
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            f.write(f"## ✅ No Plan Update Required\n\n")
            f.write("*The planner determined the current plan is still valid.*\n\n")
            f.write("---\n\n")
    
    def start_agent_conversation(self, agent_name: str, task_id: str, prompt: str):
        """Start logging a specific agent's conversation."""
        if not self.enabled:
            return
            
        self.conversation_stack.append({
            'agent': agent_name,
            'task_id': task_id,
            'start_time': datetime.datetime.now()
        })
        
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            f.write(f"## 🤖 {agent_name} Agent (Task {task_id})\n")
            f.write(f"**Start Time**: {datetime.datetime.now().strftime('%H:%M:%S')}\n\n")
            f.write(f"### Prompt Received:\n```\n{prompt}\n```\n\n")
    
    def log_agent_response(self, agent_name: str, response: str, reasoning: str = None):
        """Log an agent's response."""
        if not self.enabled:
            return
            
        # Clean up internal tool references
        cleaned_response = response
        lines = response.split('\n')
        cleaned_lines = []
        skip_next = False
        
        for i, line in enumerate(lines):
            # Skip "Tool: xxx" lines and the following "Block:" line
            if line.strip().startswith("Tool:"):
                skip_next = True
                continue
            if skip_next and line.strip().startswith("Block:"):
                skip_next = False
                continue
            skip_next = False
            cleaned_lines.append(line)
        
        cleaned_response = '\n'.join(cleaned_lines).strip()
            
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            f.write(f"### {agent_name} Response:\n")
            if reasoning:
                f.write(f"<details>\n<summary>Reasoning</summary>\n\n```\n{reasoning}\n```\n</details>\n\n")
            f.write(f"```\n{cleaned_response}\n```\n\n")
    
    def log_execution_result(self, success: bool, feedback: str = None):
        """Log the execution result of an agent's action."""
        if not self.enabled:
            return
            
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            status = "✅ Success" if success else "❌ Failed"
            f.write(f"### Execution Result: {status}\n")
            if feedback:
                f.write(f"```\n{feedback}\n```\n")
            f.write("\n")
    
    def end_agent_conversation(self, agent_name: str):
        """End logging for a specific agent's conversation."""
        if not self.enabled:
            return
            
        if self.conversation_stack and self.conversation_stack[-1]['agent'] == agent_name:
            conv_data = self.conversation_stack.pop()
            duration = datetime.datetime.now() - conv_data['start_time']
            
            with open(self.current_log_file, 'a', encoding='utf-8') as f:
                f.write(f"**Duration**: {duration.total_seconds():.2f}s\n")
                f.write("\n---\n\n")
    
    def log_final_result(self, result: str):
        """Log the final result of the entire conversation."""
        if not self.enabled:
            return
            
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            f.write(f"## 🎯 Final Result\n")
            f.write(f"**Time**: {datetime.datetime.now().strftime('%H:%M:%S')}\n\n")
            f.write(f"```\n{result}\n```\n\n")
            f.write("---\n\n")
            f.write(f"*End of conversation log*\n")
    
    def get_log_file_path(self) -> Optional[Path]:
        """Get the current log file path."""
        return self.current_log_file if self.enabled else None
    
    def disable(self):
        """Disable logging."""
        self.enabled = False
    
    def enable(self):
        """Enable logging and start a new session."""
        self.enabled = True
        self._start_new_session()


# Global instance (singleton pattern)
_conversation_logger = None

def get_conversation_logger(enabled: bool = None) -> ConversationLogger:
    """Get or create the global conversation logger instance."""
    global _conversation_logger
    if _conversation_logger is None:
        # Check config for enable status if not provided
        if enabled is None:
            import configparser
            config = configparser.ConfigParser()
            config_path = Path(__file__).parent.parent / "config.ini"
            if config_path.exists():
                config.read(config_path)
                enabled = config.getboolean('MAIN', 'conversation_logging', fallback=False)
            else:
                enabled = False
        
        _conversation_logger = ConversationLogger(enabled=enabled)
    
    return _conversation_logger