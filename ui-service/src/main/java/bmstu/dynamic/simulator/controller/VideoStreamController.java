package bmstu.dynamic.simulator.controller;

import bmstu.dynamic.simulator.config.ServicesProperties;
import bmstu.dynamic.simulator.model.VideoParams;
import bmstu.dynamic.simulator.service.VideoStreamService;
import lombok.RequiredArgsConstructor;
import org.apache.commons.io.IOUtils;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.core.io.buffer.DataBufferFactory;
import org.springframework.core.io.buffer.DefaultDataBufferFactory;
import org.springframework.http.*;
import org.springframework.http.client.ClientHttpRequest;
import org.springframework.http.server.reactive.ServerHttpResponse;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.client.OAuth2AuthorizedClient;
import org.springframework.security.oauth2.client.annotation.RegisteredOAuth2AuthorizedClient;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.server.ServerResponse;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.net.URI;
import java.util.stream.Stream;

@Controller
@RequestMapping("/video")
@RequiredArgsConstructor
public class VideoStreamController {

    private final VideoStreamService videoStreamService;
    private final ServicesProperties properties;

    @GetMapping("/video_content")
    public String videoContent() {
        return "fragments/video :: video_content";
    }

    @GetMapping("/youtube/{video_id}")
    public Mono<Void> videoByParams(@PathVariable(name = "video_id") String videoId, ServerHttpResponse response) {
        Flux<DataBuffer> dataBufferFlux = videoStreamService.getYoutubeVideoContent(videoId);

        response.getHeaders().set(HttpHeaders.CONTENT_TYPE, "multipart/x-mixed-replace; boundary=frame");
        return response.writeWith(dataBufferFlux);
    }

}