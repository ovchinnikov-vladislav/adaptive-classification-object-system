package bmstu.dynamic.simulator.client;

import org.springframework.web.bind.annotation.GetMapping;
import reactivefeign.spring.config.ReactiveFeignClient;
import reactor.core.publisher.Mono;

@ReactiveFeignClient(url = "${services.caps-visual.url}", name = "caps-visualization-client")
public interface CapsVisualizationClient {

    @GetMapping(value = "/", consumes = "text/html")
    Mono<String> getVisualize();

}
