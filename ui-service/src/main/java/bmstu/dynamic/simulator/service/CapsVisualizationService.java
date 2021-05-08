package bmstu.dynamic.simulator.service;

import bmstu.dynamic.simulator.client.CapsVisualizationClient;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cloud.openfeign.support.SpringMvcContract;
import org.springframework.stereotype.Service;
import reactivefeign.ReactiveContract;
import reactivefeign.webclient.WebReactiveFeign;
import reactor.core.publisher.Mono;

@Service
@RequiredArgsConstructor
public class CapsVisualizationService {

    private final CapsVisualizationClient client;

    public Mono<String> getVisualize() {
        return client.getVisualize();
    }

}
