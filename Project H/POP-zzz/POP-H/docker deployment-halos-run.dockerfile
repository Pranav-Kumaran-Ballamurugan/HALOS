FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

# Setup environment variables
ENV HALOS_MODE=production
ENV OPENAI_KEY=${OPENAI_KEY}
ENV ANTHROPIC_KEY=${ANTHROPIC_KEY}

# Expose API port
EXPOSE 8000

CMD ["python", "halos_server.py"]