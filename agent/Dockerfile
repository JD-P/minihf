# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app
RUN mkdir /app/weave-agent-logs/

# Copy only the requirements.txt file
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip download pip setuptools wheel --dest=package-cache
RUN pip install --no-index --find-links=package-cache pip setuptools
RUN pip download -r requirements.txt --dest=package-cache
RUN pip install -r requirements.txt --no-index --find-links=package-cache
RUN pip install django

# Install tmux and nethack
RUN apt-get clean
RUN apt-get autoclean
RUN apt-get autoremove
RUN apt-get update && apt-get upgrade -y
RUN apt-get update && apt-get install -y tmux 
RUN apt-get update && apt-get install -y nethack-console
RUN apt-get update && apt-get install -y emacs nano

# Install Firefox and geckodriver
RUN apt-get update && apt-get install -y wget
RUN apt-get update && apt-get install -y firefox-esr
RUN wget https://github.com/mozilla/geckodriver/releases/download/v0.29.1/geckodriver-v0.29.1-linux64.tar.gz
RUN tar -xvzf geckodriver-v0.29.1-linux64.tar.gz -C /usr/local/bin/
RUN chmod +x /usr/local/bin/geckodriver

# Cache HuggingFace
COPY hf_token.txt /app/
COPY cache_hf.py /app/
#ENV MODEL "weave-agent"
#ENV TOKENIZER "a-m-team/AM-Thinking-v1"
#ENV TOKENIZER "Qwen/QwQ-32B"
#ENV MODEL "mistralai/Mistral-Small-3.1-24B-Base-2503"
#ENV TOKENIZER "mistralai/Mistral-Small-3.1-24B-Base-2503"
#ENV MODEL "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
ENV TOKENIZER "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
ENV MODEL "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
#ENV TOKENIZER "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
RUN python3 cache_hf.py $TOKENIZER
RUN rm /app/hf_token.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 5001 available to the world outside this container
EXPOSE 5001

# Define environment variable
ENV NAME World

# Run the agent script when the container launches
CMD python weave_agent.py --port 5001 --bootstrap "bootstraps/nano_exercises_bootstrap.py" $MODEL --tokenizer $TOKENIZER & python -m http.server 8991 --directory "/app/weave-agent-logs/"
