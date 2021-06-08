package ru.bmstu.adapt.service;

import ru.bmstu.adapt.client.CapsVisualizationClient;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

@Service
@RequiredArgsConstructor
public class CapsVisualizationService {

    private final CapsVisualizationClient client;

    public Mono<String> getVisualize() {
        return client.getVisualize();
    }

}
