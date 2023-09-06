# Use the nvidia/cuda:11.7.1-devel-ubuntu20.04 base image
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# Set environment variables if needed (e.g., for CUDA or other configurations)
# ENV MY_VARIABLE=my_value


# basic installations
RUN apt-get update -y
RUN apt-get install python3 python3-pip -y
RUN apt-get install vim

RUN pip install --upgrade pip
#RUN pip install -r requirements.txt

RUN pip install transformers torch fastapi uvicorn pydantic





WORKDIR /DialoGPT
CMD ["python3", "model.py"]

# Update and install any additional packages or dependencies as required
# RUN apt-get update && apt-get install -y package1 package2

# Set the working directory within the container (optional)
# WORKDIR /app

# Copy your application code or files into the container
# COPY . /app

# Run any commands or scripts as needed
# CMD ["command", "arg1", "arg2"]