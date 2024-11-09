# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements.txt file
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip download -r requirements.txt --dest=package-cache
RUN pip install -r requirements.txt --no-index --find-links=package-cache
RUN pip install django

# Install Firefox and geckodriver
RUN apt-get update && apt-get install -y wget
RUN apt-get update && apt-get install -y firefox-esr
RUN wget https://github.com/mozilla/geckodriver/releases/download/v0.29.1/geckodriver-v0.29.1-linux64.tar.gz
RUN tar -xvzf geckodriver-v0.29.1-linux64.tar.gz -C /usr/local/bin/
RUN chmod +x /usr/local/bin/geckodriver

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 5001 available to the world outside this container
EXPOSE 5001

# Define environment variable
ENV NAME World

# Run the agent script when the container launches
CMD ["python", "-m", "pdb", "weave_agent.py", "--port", "5001", "--bootstrap", "bootstraps/tictactoe_bootstrap.py", "mistralai/Mixtral-8x22B-Instruct-v0.1"]
