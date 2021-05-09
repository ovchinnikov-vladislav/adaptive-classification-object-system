package bmstu.dynamic.simulator.controller;

import bmstu.dynamic.simulator.config.ServicesProperties;
import bmstu.dynamic.simulator.exception.impl.NotFoundWebException;
import bmstu.dynamic.simulator.repository.ClassificationDataService;
import bmstu.dynamic.simulator.service.VideoStreamService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.core.io.buffer.DataBufferFactory;
import org.springframework.core.io.buffer.DefaultDataBufferFactory;
import org.springframework.http.*;
import org.springframework.http.server.reactive.ServerHttpResponse;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.io.IOException;

@Controller
@RequestMapping("/video")
@RequiredArgsConstructor
@Slf4j
public class VideoStreamController {

    private final VideoStreamService videoStreamService;
    private final ServicesProperties properties;
    @Value("classpath:static/img/video-player-placeholder-very-large.png")
    private Resource videoPlaceholder;

    private final ClassificationDataService dataRepository;

    @GetMapping("/video_content")
    public String videoContent(Model model) {
        return "fragments/video :: video_content";
    }

    @GetMapping("/youtube/{video_id}")
    public Mono<Void> videoByParams(@PathVariable(name = "video_id") String videoId, ServerHttpResponse response) {
        Flux<DataBuffer> dataBufferFlux = videoStreamService.getYoutubeVideoContent(videoId);

        response.getHeaders().set(HttpHeaders.CONTENT_TYPE, "multipart/x-mixed-replace; boundary=frame");
        return response.writeWith(dataBufferFlux).onErrorResume(e -> {
            DataBufferFactory dataBufferFactory = new DefaultDataBufferFactory();
            try {
                byte[] bytes = FileUtils.readFileToByteArray(videoPlaceholder.getFile());

                DataBuffer dataBuffer = dataBufferFactory.wrap(bytes);
                response.getHeaders().set(HttpHeaders.CONTENT_TYPE, "image/png");
                return response.writeWith(Flux.just(dataBuffer));
            } catch (IOException exc) {
                throw new NotFoundWebException("not found video-placeholder");
            }
        });
    }

}