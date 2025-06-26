#!/usr/bin/env python3
"""
Test script for AgenticSeek router functionality.
This script allows testing the router's classification mechanisms and adding custom examples.
"""

import sys
import os
import argparse
import json
import configparser
from typing import List, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sources.router import AgentRouter
from sources.agents.casual_agent import CasualAgent
from sources.agents.browser_agent import BrowserAgent
from sources.agents.code_agent import CoderAgent
from sources.agents.file_agent import FileAgent
from sources.agents.planner_agent import PlannerAgent
from sources.utility import pretty_print
from sources.llm_provider import Provider

def test_single_query(router: AgentRouter, query: str, verbose: bool = False):
    """Test a single query through the router."""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    # Test complexity estimation
    complexity = router.estimate_complexity(query)
    print(f"Estimated Complexity: {complexity}")
    
    # Test individual classifiers
    if verbose:
        # Test BART pipeline
        labels = [agent.role for agent in router.agents]
        result_bart = router.pipelines['bart'](query, labels)
        print(f"\nBART Classification:")
        for label, score in zip(result_bart['labels'][:3], result_bart['scores'][:3]):
            print(f"  {label}: {score:.4f}")
        
        # Test LLM router
        predictions = router.talk_classifier.predict(query)
        predictions = [pred for pred in predictions if pred[0] not in ["HIGH", "LOW"]]
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:3]
        print(f"\nLLM Router Classification:")
        for label, score in predictions:
            print(f"  {label}: {score:.4f}")
        
        # Test voting mechanism
        print(f"\nVoting Result:")
        result = router.router_vote(query, labels, log_confidence=True)
    
    # Select agent
    selected_agent = router.select_agent(query)
    if selected_agent:
        print(f"\nSelected Agent: {selected_agent.agent_name} (type: {selected_agent.type})")
    else:
        print(f"\nNo agent selected!")

def add_custom_examples(router: AgentRouter, examples_file: str):
    """Add custom examples from a JSON file."""
    try:
        with open(examples_file, 'r') as f:
            data = json.load(f)
        
        # Add task classification examples
        if 'tasks' in data:
            texts = []
            labels = []
            for text, label in data['tasks']:
                texts.append(text)
                labels.append(label)
            router.talk_classifier.add_examples(texts, labels)
            print(f"Added {len(texts)} task classification examples")
        
        # Add complexity estimation examples
        if 'complexity' in data:
            texts = []
            labels = []
            for text, label in data['complexity']:
                texts.append(text)
                labels.append(label)
            router.complexity_classifier.add_examples(texts, labels)
            print(f"Added {len(texts)} complexity estimation examples")
            
    except Exception as e:
        print(f"Error loading examples file: {e}")

def interactive_mode(router: AgentRouter, verbose: bool = False):
    """Interactive testing mode."""
    print("\n=== Interactive Router Testing Mode ===")
    print("Type 'quit' to exit, 'verbose' to toggle detailed output")
    print("Type 'help' for more commands")
    
    while True:
        try:
            query = input("\nEnter query: ").strip()
            
            if query.lower() == 'quit':
                break
            elif query.lower() == 'verbose':
                verbose = not verbose
                print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
                continue
            elif query.lower() == 'help':
                print("\nCommands:")
                print("  quit     - Exit the program")
                print("  verbose  - Toggle detailed output")
                print("  stats    - Show router statistics")
                print("  agents   - List available agents")
                continue
            elif query.lower() == 'stats':
                print(f"\nRouter Statistics:")
                print(f"  Total agents: {len(router.agents)}")
                print(f"  Supported languages: {router.lang_analysis.languages}")
                print(f"  Query count: {router.query_count}")
                continue
            elif query.lower() == 'agents':
                print(f"\nAvailable Agents:")
                for agent in router.agents:
                    print(f"  - {agent.agent_name} (type: {agent.type}, role: {agent.role})")
                continue
            
            if query:
                test_single_query(router, query, verbose)
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def batch_test_mode(router: AgentRouter, test_file: str, verbose: bool = False):
    """Batch testing mode from a file."""
    try:
        with open(test_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        print(f"\nRunning batch test with {len(queries)} queries...")
        
        results = {
            'talk': 0,
            'web': 0,
            'code': 0,
            'files': 0,
            'planner': 0,
            'mcp': 0
        }
        
        for query in queries:
            agent = router.select_agent(query)
            if agent:
                agent_type = agent.type.replace('_agent', '')
                if agent_type in results:
                    results[agent_type] += 1
        
        print("\nBatch Test Results:")
        for agent_type, count in results.items():
            percentage = (count / len(queries)) * 100 if queries else 0
            print(f"  {agent_type}: {count} ({percentage:.1f}%)")
            
    except Exception as e:
        print(f"Error in batch test: {e}")

def main():
    parser = argparse.ArgumentParser(description='Test AgenticSeek router functionality')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-q', '--query', type=str, help='Test a single query')
    parser.add_argument('-b', '--batch', type=str, help='Batch test from file')
    parser.add_argument('-e', '--examples', type=str, help='Add custom examples from JSON file')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode (default)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    # Initialize provider from config
    provider_name = config.get('MAIN', 'provider_name', fallback='ollama')
    model = config.get('MAIN', 'provider_model', fallback='llama3:latest')
    server = config.get('MAIN', 'provider_server_address', fallback='127.0.0.1:11434')
    is_local = config.getboolean('MAIN', 'is_local', fallback=True)
    
    print(f"Initializing provider: {provider_name} with model {model}...")
    provider = Provider(provider_name, model, server, is_local)
    
    # Initialize agents
    print("Initializing agents and router...")
    agents = [
        CasualAgent("Casual", "prompts/base/casual_agent.txt", provider),
        BrowserAgent("Web", "prompts/base/browser_agent.txt", provider),
        CoderAgent("Coder", "prompts/base/coder_agent.txt", provider),
        FileAgent("File", "prompts/base/file_agent.txt", provider),
        PlannerAgent("Planner", "prompts/base/planner_agent.txt", provider)
    ]
    
    # Initialize router
    router = AgentRouter(agents)
    
    # Add custom examples if provided
    if args.examples:
        add_custom_examples(router, args.examples)
    
    # Execute based on mode
    if args.query:
        test_single_query(router, args.query, args.verbose)
    elif args.batch:
        batch_test_mode(router, args.batch, args.verbose)
    else:
        # Default to interactive mode
        interactive_mode(router, args.verbose)

if __name__ == "__main__":
    main()