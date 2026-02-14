FROM python:3
WORKDIR /app

COPY requirements.txt ./ 
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./main.py
COPY model ./model

CMD [ "fastapi", "run", "./main.py"]
