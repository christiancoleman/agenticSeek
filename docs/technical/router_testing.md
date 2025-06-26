# Router Testing and Training Guide

This guide explains how to test and improve the AgenticSeek router's classification capabilities.

## Overview

The AgenticSeek router uses a dual classification system:
1. **BART Zero-shot Classifier** - Pre-trained model from HuggingFace
2. **Adaptive LLM Router** - Custom DistilBERT-based classifier with few-shot learning

These work together through a voting mechanism to select the appropriate agent.

## Testing the Router

### 1. Direct Router Testing

You can test the router directly by running it as a standalone module:

```bash
cd /mnt/c/Utils/agenticSeek
python sources/router.py
```

This runs predefined test queries and shows which agent is selected.

### 2. Using the Test Script

A comprehensive test script is available at `test_router.py`:

```bash
# Interactive mode (default)
python test_router.py

# Test a single query with verbose output
python test_router.py -q "Create a web scraper in Python" -v

# Batch test from file
python test_router.py -b test_queries.txt

# Add custom examples before testing
python test_router.py -e test_examples.json -i
```

### 3. Test Script Features

The test script provides several modes:

- **Interactive Mode**: Test queries one at a time with immediate feedback
- **Verbose Mode**: See detailed classification scores from both BART and LLM router
- **Batch Mode**: Test multiple queries from a file and see statistics
- **Custom Examples**: Add your own training examples before testing

### Interactive Commands:
- `quit` - Exit the program
- `verbose` - Toggle detailed output
- `stats` - Show router statistics
- `agents` - List available agents

## Understanding Router Classification

### Agent Types and Roles

1. **CasualAgent** (role: "talk") - General conversation
2. **BrowserAgent** (role: "web") - Web searches and browsing
3. **CoderAgent** (role: "code") - Programming and debugging
4. **FileAgent** (role: "files") - File system operations
5. **PlannerAgent** (role: "planner") - Complex multi-step tasks

### Complexity Estimation

The router also estimates task complexity:
- **LOW**: Simple, single-step tasks
- **HIGH**: Complex, multi-step tasks (routed to PlannerAgent)

## Adding Training Examples

### 1. Through Code (Permanent)

Edit `sources/router.py` and add examples to the `learn_few_shots_tasks()` or `learn_few_shots_complexity()` methods:

```python
def learn_few_shots_tasks(self) -> None:
    few_shots = [
        # Add your examples here
        ("Your query text", "agent_role"),  # agent_role: talk, web, code, files, mcp
        # ...
    ]
```

### 2. Through JSON File (Runtime)

Create a JSON file with your examples:

```json
{
  "tasks": [
    ["Query text", "agent_role"],
    ["Another query", "another_role"]
  ],
  "complexity": [
    ["Simple task", "LOW"],
    ["Complex multi-step task", "HIGH"]
  ]
}
```

Then load it:
```bash
python test_router.py -e your_examples.json
```

### 3. Example Format Guidelines

For task classification:
- "talk" - Casual conversation, questions about the AI, jokes, stories
- "web" - Web searches, online research, finding information
- "code" - Writing code, debugging, programming tasks
- "files" - Finding files, organizing folders, file operations
- "mcp" - MCP protocol operations (experimental)

For complexity estimation:
- "LOW" - Single clear action, straightforward task
- "HIGH" - Multiple steps, requires planning, combines different operations

## Testing Strategies

### 1. Test Edge Cases

Test queries that might confuse the router:
- Ambiguous queries: "Tell me about Python" (talk or web?)
- Multi-part queries: "Find my resume and upload it to LinkedIn"
- Short queries: "hi", "help", "fix this"

### 2. Test Language Support

The router supports multiple languages (configurable):
```
"你好" (Chinese)
"Bonjour" (French)
"Write code in español"
```

### 3. Performance Metrics

When batch testing, look for:
- Accuracy: Are queries routed to the correct agent?
- Consistency: Does the same query always route the same way?
- Language handling: Are non-English queries handled correctly?

## Advanced Testing

### 1. Modify Voting Weights

The router uses confidence scores from both classifiers. You can log these to understand decisions:

```python
router.router_vote(query, labels, log_confidence=True)
```

### 2. Test Individual Components

Test just the BART classifier:
```python
result = router.pipelines['bart'](query, labels)
```

Test just the LLM router:
```python
predictions = router.talk_classifier.predict(query)
```

### 3. Analyze Misclassifications

When the router selects the wrong agent:
1. Check both classifier outputs in verbose mode
2. See which classifier was wrong
3. Add corrective examples to improve accuracy

## Training Data Management

### Current Training Data

The router is pre-trained with:
- ~200 task classification examples
- ~200 complexity estimation examples
- Balanced across all agent types
- Multi-language examples

### Adding New Training Data

When adding examples:
1. Keep balance across agent types
2. Include edge cases and ambiguous queries
3. Add both positive and negative examples
4. Test after adding to ensure improvement

### Example Selection Tips

Good examples are:
- Clear and unambiguous for their category
- Representative of real user queries
- Diverse in phrasing and structure
- Include domain-specific terminology

## Troubleshooting

### Common Issues

1. **Router always selects the same agent**
   - Check if one classifier is dominating
   - Verify examples are balanced

2. **Poor performance on new query types**
   - Add specific examples for those queries
   - Check if complexity estimation is interfering

3. **Language detection issues**
   - Verify language is in supported list
   - Check translation is working correctly

### Debug Mode

Enable debug logging in the router:
```python
router.logger.info(f"Debug info: {variable}")
```

## Best Practices

1. **Regular Testing**: Test the router with new query types regularly
2. **Incremental Training**: Add examples gradually and test impact
3. **Version Control**: Track changes to training examples
4. **Documentation**: Document why specific examples were added
5. **User Feedback**: Collect misrouted queries from users for training