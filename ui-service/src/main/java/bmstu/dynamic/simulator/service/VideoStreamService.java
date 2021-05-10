package bmstu.dynamic.simulator.service;

import bmstu.dynamic.simulator.config.ServicesProperties;
import bmstu.dynamic.simulator.exception.impl.NotFoundWebException;
import bmstu.dynamic.simulator.model.DetectionObject;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.core.io.Resource;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.core.io.buffer.DataBufferFactory;
import org.springframework.core.io.buffer.DefaultDataBufferFactory;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static bmstu.dynamic.simulator.config.VideoStreamConstants.ACCEPT_RANGES;
import static bmstu.dynamic.simulator.config.VideoStreamConstants.BYTES;
import static bmstu.dynamic.simulator.config.VideoStreamConstants.BYTE_RANGE;
import static bmstu.dynamic.simulator.config.VideoStreamConstants.CONTENT_LENGTH;
import static bmstu.dynamic.simulator.config.VideoStreamConstants.CONTENT_RANGE;
import static bmstu.dynamic.simulator.config.VideoStreamConstants.CONTENT_TYPE;
import static bmstu.dynamic.simulator.config.VideoStreamConstants.VIDEO;
import static bmstu.dynamic.simulator.config.VideoStreamConstants.VIDEO_CONTENT;

@Slf4j
@Service
@RequiredArgsConstructor
public class VideoStreamService {

    private final ServicesProperties properties;

    public Flux<DataBuffer> getYoutubeVideoContent(String videoId, String userId, String uuidSession) {
        WebClient client = WebClient.create(properties.getMachineLearningUrl());

        return client.get()
                .uri("/youtube_video_feed/" + videoId + '/' + userId + '/' + uuidSession)
                .accept(MediaType.TEXT_HTML)
                .retrieve()
                .bodyToFlux(DataBuffer.class);
    }

    public Flux<DataBuffer> getCameraVideoContent(String videoAddr, String userId, String uuidSession) {
        WebClient client = WebClient.create(properties.getMachineLearningUrl());

        return client.get()
                .uri("/camera_video_feed/" + userId + '/' + uuidSession + "?video_addr=" + videoAddr)
                .accept(MediaType.TEXT_HTML)
                .retrieve()
                .bodyToFlux(DataBuffer.class);
    }

    public Flux<Map> getDetectionObjects(String userId, String uuidSession) {
        WebClient client = WebClient.create(properties.getStatUrl());

        return client.get()
                .uri("/detection-objects/" + userId + "/" + uuidSession)
                .accept(MediaType.APPLICATION_JSON)
                .retrieve()
                .bodyToFlux(Map.class);
    }

    public Flux<Map> getDetectionObjectsByNumObject(String userId, String uuidSession, Integer numObject) {
        WebClient client = WebClient.create(properties.getStatUrl());

        return client.get()
                .uri("/detection-objects/" + userId + "/" + uuidSession + "/" + numObject)
                .accept(MediaType.APPLICATION_JSON)
                .retrieve()
                .bodyToFlux(Map.class);
    }

}