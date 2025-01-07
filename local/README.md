## Local Environment Requirements
- Python 3.11.5
- Install the required packages using the following command:
    ```bash
    pip install -r requirements.txt
    ```
- Install Ollama from [here](https://ollama.com/download)

## Ollama Requirements
- LLaMA 3.1 model recommended. Codebase done with `llama3.1:8b`.

## Ejecucion
### Embedir documentos
Para embeber los documentos PDF, se debe ejecutar el siguiente comando:
```bash
python embed_documents.py
```
> Nota: Este comando puede tardar varios minutos en ejecutarse. Documentos deben estar en la carpeta `data/testing_docs`.

### Ejecutar el chat
Para ejecutar el chat, se debe ejecutar el siguiente comando:
```bash
python chat.py
```

### Notas
Si un solo documento no es relevante, el agente ignora todos (corregir)