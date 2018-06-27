FROM ubuntu:18.04

RUN apt-get -qq -y update \
 && apt-get -qq -y install curl gnupg

RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list \
 && curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add - \
 && apt-get -qq -y update \
 && apt-get -qq -y install tensorflow-model-server \
 && apt-get -qq clean \
 && rm -rf /var/lib/apt/lists/*

# gRPC port
EXPOSE 8050
# REST API port
EXPOSE 8051

CMD tensorflow_model_server \
  --port=8500 --rest_api_port=8051 \
  --model_name="$MODEL_NAME" \
  --model_base_path="$MODEL_BASE_PATH"
