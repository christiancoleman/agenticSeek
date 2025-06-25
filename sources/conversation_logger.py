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
        self.session_folder = None
        self.current_outline_file = None
        self.current_transcript_file = None
        self.conversation_stack = []
        self.api_sequence_number = 0
        
        if self.enabled:
            self.log_dir.mkdir(exist_ok=True)
            self._start_new_session()
    
    def _get_next_session_id(self) -> str:
        """Get the next session ID by checking existing folders."""
        existing_ids = []
        if self.log_dir.exists():
            for folder in self.log_dir.iterdir():
                if folder.is_dir() and folder.name.startswith("ID"):
                    try:
                        # Extract the numeric part
                        num = int(folder.name[2:])
                        existing_ids.append(num)
                    except ValueError:
                        continue
        
        next_id = max(existing_ids) + 1 if existing_ids else 1
        return f"ID{next_id:05d}"
    
    def _start_new_session(self):
        """Start a new conversation logging session."""
        self.current_session_id = self._get_next_session_id()
        self.session_folder = self.log_dir / self.current_session_id
        self.session_folder.mkdir(exist_ok=True)
        
        # Reset sequence number for new session
        self.api_sequence_number = 0
        
        # Set file paths
        self.current_outline_file = self.session_folder / f"{self.current_session_id}_outline.txt"
        self.current_transcript_file = self.session_folder / f"{self.current_session_id}_transcript.txt"
        
        # Write header for outline file
        with open(self.current_outline_file, 'w', encoding='utf-8') as f:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            f.write(f"[LOGGER: Conversation Outline - Session {self.current_session_id} - {timestamp}]\n\n")
            f.write("[LOGGER: This log shows the sequential flow of agent conversations.]\n\n")
            f.write("[LOGGER: =====================================]\n\n")
            
        # Write header for transcript file
        with open(self.current_transcript_file, 'w', encoding='utf-8') as f:
            f.write(f"[LOGGER: Raw Transcript - Session {self.current_session_id}]\n\n")
            f.write("[LOGGER: This log shows the exact messages sent to and received from the LLM.]\n")
            f.write("[LOGGER: Everything prefixed with [LOGGER:] is added by the logging system.]\n")
            f.write("[LOGGER: Everything else is the actual content sent to or received from the LLM.]\n\n")
            f.write("[LOGGER: =====================================]\n\n")
    
    def log_user_query(self, query: str):
        """Log the initial user query."""
        if not self.enabled:
            return
            
        with open(self.current_outline_file, 'a', encoding='utf-8') as f:
            f.write(f"[LOGGER: 🧑 User Query]\n")
            f.write(f"[LOGGER: Time: {datetime.datetime.now().strftime('%H:%M:%S')}]\n\n")
            f.write(f"{query}\n\n")
            f.write("[LOGGER: ---]\n\n")
    
    def log_router_decision(self, agent_type: str, complexity: str = None):
        """Log the router's decision."""
        if not self.enabled:
            return
            
        with open(self.current_outline_file, 'a', encoding='utf-8') as f:
            f.write(f"[LOGGER: 🚦 Router Decision]\n")
            f.write(f"[LOGGER: Selected Agent: {agent_type}]\n")
            if complexity:
                f.write(f"[LOGGER: Complexity: {complexity}]\n")
            f.write("\n[LOGGER: ---]\n\n")
    
    def log_planner_plan(self, plan: list, is_update: bool = False):
        """Log the planner's task breakdown."""
        if not self.enabled:
            return
            
        with open(self.current_outline_file, 'a', encoding='utf-8') as f:
            if is_update:
                f.write(f"[LOGGER: 🔄 Updated Plan]\n\n")
                f.write("[LOGGER: The planner has revised the task breakdown:]\n\n")
            else:
                f.write(f"[LOGGER: 📋 Planner's Task Breakdown]\n\n")
                
            for i, task_item in enumerate(plan, 1):
                # Handle both list format [task_name, task_dict] and direct dict format
                if isinstance(task_item, list) and len(task_item) >= 2:
                    task_name, task = task_item[0], task_item[1]
                    f.write(f"[LOGGER: Task {i}: {task_name}]\n")
                    f.write(f"[LOGGER: Agent: {task.get('agent', 'Unknown')}]\n")
                    f.write(f"[LOGGER: ID: {task.get('id', 'N/A')}]\n")
                    f.write(f"[LOGGER: Dependencies: {task.get('need', [])}]\n")
                    f.write(f"[LOGGER: Description:] {task.get('task', 'No description')}\n\n")
                elif isinstance(task_item, dict):
                    # Direct dict format
                    f.write(f"[LOGGER: Task {i}: {task_item.get('agent', 'Unknown')}]\n")
                    f.write(f"[LOGGER: ID: {task_item.get('id', 'N/A')}]\n")
                    f.write(f"[LOGGER: Dependencies: {task_item.get('need', [])}]\n")
                    f.write(f"[LOGGER: Description:] {task_item.get('task', 'No description')}\n\n")
            f.write("[LOGGER: ---]\n\n")
    
    def log_planner_error(self, error_msg: str, raw_response: str = None):
        """Log planner errors and retry attempts."""
        if not self.enabled:
            return
            
        with open(self.current_outline_file, 'a', encoding='utf-8') as f:
            f.write(f"[LOGGER: ⚠️ Planner Error]\n\n")
            f.write(f"[LOGGER: Error: {error_msg}]\n\n")
            if raw_response:
                f.write(f"[LOGGER: Raw LLM Response:]\n{raw_response}\n\n")
            f.write("[LOGGER: Retrying...]\n\n")
            f.write("[LOGGER: ---]\n\n")
    
    def log_planner_reasoning(self, reasoning: str):
        """Log the planner's reasoning before the task breakdown."""
        if not self.enabled:
            return
            
        with open(self.current_outline_file, 'a', encoding='utf-8') as f:
            f.write(f"[LOGGER: 🤔 Planner's Reasoning]\n\n")
            f.write(f"{reasoning}\n\n")
            f.write("[LOGGER: ---]\n\n")
    
    def log_no_update_decision(self):
        """Log when planner decides no update is needed."""
        if not self.enabled:
            return
            
        with open(self.current_outline_file, 'a', encoding='utf-8') as f:
            f.write(f"[LOGGER: ✅ No Plan Update Required]\n\n")
            f.write("[LOGGER: The planner determined the current plan is still valid.]\n\n")
            f.write("[LOGGER: ---]\n\n")
    
    def start_agent_conversation(self, agent_name: str, task_id: str, prompt: str):
        """Start logging a specific agent's conversation."""
        if not self.enabled:
            return
            
        self.conversation_stack.append({
            'agent': agent_name,
            'task_id': task_id,
            'start_time': datetime.datetime.now()
        })
        
        with open(self.current_outline_file, 'a', encoding='utf-8') as f:
            f.write(f"[LOGGER: 🤖 {agent_name} Agent (Task {task_id})]\n")
            f.write(f"[LOGGER: Start Time: {datetime.datetime.now().strftime('%H:%M:%S')}]\n\n")
            f.write(f"[LOGGER: Prompt Received:]\n{prompt}\n\n")
    
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
            
        with open(self.current_outline_file, 'a', encoding='utf-8') as f:
            f.write(f"[LOGGER: {agent_name} Response:]\n")
            if reasoning:
                f.write(f"[LOGGER: Reasoning:]\n{reasoning}\n\n")
            f.write(f"{cleaned_response}\n\n")
    
    def log_execution_result(self, success: bool, feedback: str = None):
        """Log the execution result of an agent's action."""
        if not self.enabled:
            return
            
        with open(self.current_outline_file, 'a', encoding='utf-8') as f:
            status = "✅ Success" if success else "❌ Failed"
            f.write(f"[LOGGER: Execution Result: {status}]\n")
            if feedback:
                f.write(f"{feedback}\n")
            f.write("\n")
    
    def end_agent_conversation(self, agent_name: str):
        """End logging for a specific agent's conversation."""
        if not self.enabled:
            return
            
        if self.conversation_stack and self.conversation_stack[-1]['agent'] == agent_name:
            conv_data = self.conversation_stack.pop()
            duration = datetime.datetime.now() - conv_data['start_time']
            
            with open(self.current_outline_file, 'a', encoding='utf-8') as f:
                f.write(f"[LOGGER: Duration: {duration.total_seconds():.2f}s]\n")
                f.write("\n[LOGGER: ---]\n\n")
    
    def log_final_result(self, result: str):
        """Log the final result of the entire conversation."""
        if not self.enabled:
            return
            
        with open(self.current_outline_file, 'a', encoding='utf-8') as f:
            f.write(f"[LOGGER: 🎯 Final Result]\n")
            f.write(f"[LOGGER: Time: {datetime.datetime.now().strftime('%H:%M:%S')}]\n\n")
            f.write(f"{result}\n\n")
            f.write("[LOGGER: ---]\n\n")
            f.write(f"[LOGGER: End of conversation log]\n")
    
    def log_llm_interaction(self, agent_name: str, messages: list, response: str, model: str = None):
        """Log raw LLM interactions to transcript file."""
        if not self.enabled:
            return
            
        with open(self.current_transcript_file, 'a', encoding='utf-8') as f:
            f.write(f"[LOGGER: LLM Interaction - {agent_name} at {datetime.datetime.now().strftime('%H:%M:%S')}]\n")
            if model:
                f.write(f"[LOGGER: Model - {model}]\n")
            f.write("[LOGGER: BEGIN MESSAGES SENT TO LLM]\n\n")
            
            # Log the full message history
            for msg in messages:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                f.write(f"[LOGGER: ---{role.upper()} MESSAGE START---]\n")
                f.write(content)
                if not content.endswith('\n'):
                    f.write('\n')
                f.write(f"[LOGGER: ---{role.upper()} MESSAGE END---]\n\n")
            
            f.write("[LOGGER: END MESSAGES SENT TO LLM]\n")
            f.write("[LOGGER: BEGIN LLM RESPONSE]\n\n")
            f.write("[LOGGER: ---RAW LLM RESPONSE START---]\n")
            f.write(response)
            if not response.endswith('\n'):
                f.write('\n')
            f.write("[LOGGER: ---RAW LLM RESPONSE END---]\n\n")
            f.write("[LOGGER: =====================================]\n\n")
    
    def log_session_separator(self):
        """Log a separator between conversations in the same session."""
        if not self.enabled:
            return
            
        separator = "="*80
        with open(self.current_outline_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[LOGGER: {separator}]\n")
            f.write("[LOGGER: 🔄 New Conversation in Session]\n")
            f.write(f"[LOGGER: Time: {datetime.datetime.now().strftime('%H:%M:%S')}]\n")
            f.write(f"[LOGGER: {separator}]\n\n")
            
        with open(self.current_transcript_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[LOGGER: {separator}]\n")
            f.write("[LOGGER: 🔄 New Conversation in Session]\n")
            f.write(f"[LOGGER: Time: {datetime.datetime.now().strftime('%H:%M:%S')}]\n")
            f.write(f"[LOGGER: {separator}]\n\n")
    
    def log_api_request(self, agent_name: str, provider: str, url: str, headers: dict, body: dict):
        """Log an API request to a JSON file."""
        if not self.enabled:
            return
            
        self.api_sequence_number += 1
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.current_session_id}_{self.api_sequence_number:03d}_req_{timestamp}_{agent_name}.json"
        filepath = self.session_folder / filename
        
        # Redact sensitive headers
        safe_headers = headers.copy() if headers else {}
        for key in ['Authorization', 'X-API-Key', 'api-key', 'x-api-key']:
            if key in safe_headers:
                safe_headers[key] = "REDACTED"
        
        request_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "sequence": self.api_sequence_number,
            "agent": agent_name,
            "provider": provider,
            "url": url,
            "headers": safe_headers,
            "body": body
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(request_data, f, indent=2, ensure_ascii=False)
    
    def log_api_response(self, agent_name: str, provider: str, status_code: int, headers: dict, body: dict, duration_ms: float = None):
        """Log an API response to a JSON file."""
        if not self.enabled:
            return
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.current_session_id}_{self.api_sequence_number:03d}_res_{timestamp}_{agent_name}.json"
        filepath = self.session_folder / filename
        
        response_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "sequence": self.api_sequence_number,
            "agent": agent_name,
            "provider": provider,
            "status_code": status_code,
            "headers": dict(headers) if headers else {},
            "body": body,
            "duration_ms": duration_ms
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=2, ensure_ascii=False)
    
    def get_log_file_path(self) -> Optional[Path]:
        """Get the current log file path."""
        return self.current_outline_file if self.enabled else None
    
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