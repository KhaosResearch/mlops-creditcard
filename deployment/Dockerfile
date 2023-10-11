FROM python:3.11.3

LABEL org.opencontainers.image.source=https://github.com/KhaosResearch/mlops-creditcard/
LABEL org.opencontainers.image.description="Creditcard model deployment image"
LABEL org.opencontainers.image.licenses=MIT

WORKDIR /home
COPY ./CreditcardSeldonWrapper.py /home
COPY ./requirements.txt /home
RUN chown -R 8888 /home
RUN chmod -R 755 /home

RUN pip install --upgrade pip

RUN pip install -r requirements.txt
# Port for GRPC
EXPOSE 5000
# Port for REST
EXPOSE 9000

# Fixed value for required mlflow and seldon environment variables (don't modify)
ENV MODEL_NAME="CreditcardSeldonWrapper"
ENV SERVICE_TYPE="MODEL"

# Default value for required mlflow and seldon environment variables
ENV MLFLOW_S3_IGNORE_TLS=""
ENV MLFLOW_S3_ENDPOINT_URL=""
ENV MLFLOW_TRACKING_URI=""
ENV AWS_ACCESS_KEY_ID=""
ENV AWS_SECRET_ACCESS_KEY=""

CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE
