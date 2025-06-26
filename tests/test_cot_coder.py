#!/usr/bin/env python3
"""
Test script to compare regular Coder Agent vs Chain-of-Thought Coder Agent
"""

import sys
import os
import argparse
import configparser

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sources.agents.code_agent import CoderAgent
from sources.llm_provider import Provider
from sources.utility import pretty_print


async def test_coder_agent(prompt_file: str, task: str, agent_name: str):
    """Test a coder agent with a specific prompt file."""
    
    # Load configuration
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    # Initialize provider
    provider_name = config.get('MAIN', 'provider_name', fallback='ollama')
    model = config.get('MAIN', 'provider_model', fallback='llama3:latest')
    server = config.get('MAIN', 'provider_server_address', fallback='127.0.0.1:11434')
    is_local = config.getboolean('MAIN', 'is_local', fallback=True)
    
    print(f"\n{'='*60}")
    print(f"Testing {agent_name}")
    print(f"Prompt file: {prompt_file}")
    print(f"{'='*60}\n")
    
    # Initialize provider and agent
    provider = Provider(provider_name, model, server, is_local)
    agent = CoderAgent(agent_name, prompt_file, provider)
    
    # Process the task
    print(f"Task: {task}\n")
    print("Agent is processing...\n")
    
    result, reasoning = await agent.process(task, None)
    
    print("\n--- Agent Response ---")
    agent.show_answer()
    
    if reasoning:
        print("\n--- Reasoning ---")
        print(reasoning)
    
    return result


async def main():
    parser = argparse.ArgumentParser(description='Test Chain-of-Thought vs Regular Coder Agent')
    parser.add_argument('-t', '--task', type=str, 
                       default="Create a simple calculator with basic operations (+, -, *, /)",
                       help='Task to give to the coder agent')
    parser.add_argument('--cot-only', action='store_true', 
                       help='Only test the CoT version')
    parser.add_argument('--regular-only', action='store_true', 
                       help='Only test the regular version')
    
    args = parser.parse_args()
    
    # Test regular coder agent
    if not args.cot_only:
        await test_coder_agent(
            "prompts/base/coder_agent.txt",
            args.task,
            "Regular Coder"
        )
    
    # Test CoT coder agent
    if not args.regular_only:
        await test_coder_agent(
            "prompts/base/coder_agent_cot.txt",
            args.task,
            "CoT Coder"
        )
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())