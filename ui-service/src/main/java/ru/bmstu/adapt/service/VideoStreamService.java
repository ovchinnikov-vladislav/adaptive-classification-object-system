package ru.bmstu.adapt.service;

import ru.bmstu.adapt.config.ServicesProperties;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;

import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;

import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class VideoStreamService {

    private final ServicesProperties properties;

    public Flux<DataBuffer> getYoutubeVideoContent(String videoId, String userId, String uuidSession) {
        WebClient client = WebClient.create(properties.getMachineLearningUrl());

        return client.get()
                .uri("/youtube_video/" + videoId + '/' + userId + '/' + uuidSession)
                .accept(MediaType.TEXT_HTML)
                .retrieve()
                .bodyToFlux(DataBuffer.class);
    }

    public Flux<DataBuffer> getCameraVideoContent(String videoAddr, String userId, String uuidSession) {
        WebClient client = WebClient.create(properties.getMachineLearningUrl());

        return client.get()
                .uri("/camera_video/" + userId + '/' + uuidSession + "?video_addr=" + videoAddr)
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