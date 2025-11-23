FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy everything from your project into the image
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# By default run DQN training when the container starts
CMD ["python", "-m", "src.train_dqn", "--config", "config.yaml"]
