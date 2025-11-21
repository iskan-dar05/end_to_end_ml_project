FROM python:3.9-slim

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

COPY dataset .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files
COPY . .

EXPOSE 8888


# Run EDA.py for testing
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]