import os, sys
import re
from io import StringIO
import subprocess

if __name__ == "__main__": # if running as a script for individual testing
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sources.tools.tools import Tools
from sources.tools.safety import is_any_unsafe

class HTMLInterpreter(Tools):
    """
    This class is a tool to allow agent for HTML preservation.
    """
    def __init__(self):
        super().__init__()
        self.tag = "html"
        self.name = "HTML Interpreter"
        self.description = "This tool allows the agent to preserve HTML content."
    
    def execute(self, blocks: [str], safety=False):
        """
        Execute HTMLInterpreter.
        """
        print("Executing HTML interpreter's execute method...")
        
        return "concat_output"  # Placeholder for concatenated output
    
    def interpreter_feedback(self, output):
        """
        Provide feedback to the AI based on the HTML execution output.
        """
        if self.execution_failure_check(output):
            feedback = f"[failure] HTML execution error:\n{output}"
        else:
            feedback = "[success] HTML executed successfully:\n" + output
        return feedback
    
    def execution_failure_check(self, output):
        """
        Check if HTML execution failed based on the output.
        """
        error_indicators = [
            "error:",
            "Error:",
            "ERROR:",
            "SyntaxError",
            "ReferenceError",
            "TypeError", 
            "RangeError",
            "EvalError",
            "URIError",
            "execution failed",
            "not installed",
            "not in PATH",
            "timed out"
        ]
        
        output_lower = output.lower()
        for indicator in error_indicators:
            if indicator.lower() in output_lower:
                return True
        
        return False

if __name__ == "__main__":
    # Test the JavaScript interpreter
    html = HTMLInterpreter()
    html.work_dir = os.getcwd()
    
    # Test code
    test_code = '''
<html>
<head>
	<title>Test HTML</title>
    </head>
<body>	
</body>
</html>
'''
    
    print("Testing HTML interpreter...")
    output = html.execute([test_code])
    print(output)
    print("\nFeedback:", html.interpreter_feedback(output))