package ru.bmstu.adapt.client;

import org.springframework.web.bind.annotation.GetMapping;
import reactivefeign.spring.config.ReactiveFeignClient;
import reactor.core.publisher.Mono;

public interface CapsVisualizationClient {

    @GetMapping(value = "/", consumes = "text/html")
    Mono<String> getVisualize();

}
