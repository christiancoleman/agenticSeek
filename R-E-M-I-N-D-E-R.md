# Choose LLM provider: Ollama or LM-Studio

## Starting ollama
1. $env:OLLAMA_HOST="0.0.0.0:11434"
2. ollama serve
3. Change config.ini

```
provider_name = ollama
provider_model = deepseek-r1:14b
provider_server_address = host.docker.internal:11434
```

## Starting LM-Studio
1. Just open 'er up
2. Change config.ini

```
provider_name = lm-studio
provider_model = deepseek-r1-distill-qwen-14b-abliterated-v2
provider_server_address = host.docker.internal:1234
```

# Starting everything else
1. start .\start_services.cmd full