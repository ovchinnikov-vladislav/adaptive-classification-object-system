package bmstu.dynamic.simulator.service;

import bmstu.dynamic.simulator.client.CapsVisualizationClient;
import org.springframework.cloud.openfeign.support.SpringMvcContract;
import org.springframework.stereotype.Service;
import reactivefeign.ReactiveContract;
import reactivefeign.webclient.WebReactiveFeign;
import reactor.core.publisher.Mono;

@Service
public class CapsVisualizationService {

    private final CapsVisualizationClient client = WebReactiveFeign.<CapsVisualizationClient>builder()
            .contract(new ReactiveContract(new SpringMvcContract()))
            .target(CapsVisualizationClient.class, "http://localhost:8081");

    public Mono<String> getVisualize() {
        return client.getVisualize();
    }

}
