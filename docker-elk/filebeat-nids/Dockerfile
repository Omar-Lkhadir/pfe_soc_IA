ARG ELASTIC_VERSION

FROM docker.elastic.co/beats/filebeat:${ELASTIC_VERSION:-9.3.1}

USER root
COPY config/filebeat.yml /usr/share/filebeat/filebeat.yml
RUN chown root:filebeat /usr/share/filebeat/filebeat.yml && chmod go-w /usr/share/filebeat/filebeat.yml
USER filebeat
