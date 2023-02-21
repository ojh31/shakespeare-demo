# Create environment
FROM mambaorg/micromamba:1.3.1
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
RUN micromamba install --yes --file /tmp/env.yaml && \
    micromamba clean --all --yes

# Run app
COPY . /app/
WORKDIR /app/
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN python app.py