# TimeWarp web environments: Wiki, News and Shop (WebShop), each in six UI versions.
#
# The environment code is taken from the upstream TimeWarp repository at a pinned
# commit, and the environment data from the `timewarp-env-data` Hugging Face
# dataset at a pinned revision, so every build serves the same pages.
FROM python:3.11-slim-trixie@sha256:bab1b7ef4b450c81002278d035eff85ebe394ae94df904f7a3ba14f7e16e487b

ARG TIMEWARP_COMMIT=4978e690ef2ad33d81a4a0c043da3f80588902a0
ARG ENV_DATA_REVISION=bfed490a8b4b5dae94b126be6105b84445dcbc4b
ARG ENV_DATA_URL=https://huggingface.co/datasets/sparklabutah/timewarp-env-data/resolve/${ENV_DATA_REVISION}

# Pyserini (WebShop search) runs Anserini on the JVM and needs Java 21.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s "$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")" /opt/java
ENV JAVA_HOME=/opt/java

COPY requirements-x86_64.txt requirements-aarch64.txt /tmp/
RUN pip install --no-cache-dir --require-hashes --no-deps \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r "/tmp/requirements-$(uname -m).txt" \
    && rm /tmp/requirements-*.txt

WORKDIR /opt/timewarp
RUN curl -fsSL "https://github.com/sparklabutah/timewarp/archive/${TIMEWARP_COMMIT}.tar.gz" -o /tmp/timewarp.tar.gz \
    && tar -xzf /tmp/timewarp.tar.gz --strip-components=1 "timewarp-${TIMEWARP_COMMIT}/env" \
    && rm /tmp/timewarp.tar.gz

RUN curl -fsSL "${ENV_DATA_URL}/wiki_index.pkl" -o env/wiki/wiki_index.pkl \
    && curl -fsSL "${ENV_DATA_URL}/news_index.pkl" -o env/news/news_index.pkl \
    && mkdir -p env/webshop/data \
    && for file in items_shuffle_1000.json items_ins_v2_1000.json items_human_ins.json; do \
        curl -fsSL "${ENV_DATA_URL}/webshop/${file}" -o "env/webshop/data/${file}"; \
    done

# Build the WebShop search index (mirrors env/webshop/setup.sh).
RUN cd env/webshop/search_engine \
    && mkdir -p resources resources_100 resources_1k resources_100k indexes \
    && python convert_product_file_format.py \
    && python -m pyserini.index.lucene --collection JsonCollection --input resources \
        --index indexes --generator DefaultLuceneDocumentGenerator --threads 1 \
        --storePositions --storeDocvectors --storeRaw

COPY start_envs.sh healthcheck.py /opt/timewarp/
RUN chmod 0755 /opt/timewarp/start_envs.sh

# The servers only read from /opt/timewarp, so they run as an unprivileged user.
RUN useradd --create-home --shell /usr/sbin/nologin timewarp
USER timewarp

ENTRYPOINT ["/opt/timewarp/start_envs.sh"]
