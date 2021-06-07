package bmstu.dynamic.simulator.config;

import bmstu.dynamic.simulator.client.CapsVisualizationClient;
import bmstu.dynamic.simulator.client.MachineLearningClient;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cloud.openfeign.support.SpringMvcContract;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.ExchangeStrategies;
import org.springframework.web.reactive.function.client.WebClient;
import reactivefeign.ReactiveContract;
import reactivefeign.spring.config.EnableReactiveFeignClients;
import reactivefeign.webclient.WebReactiveFeign;

@Configuration
@EnableReactiveFeignClients
@RequiredArgsConstructor
public class WebFluxConfig {

    private final ServicesProperties servicesProperties;

    @Bean
    public CapsVisualizationClient capsVisualizationClient() {
        return WebReactiveFeign.<CapsVisualizationClient>builder()
                .contract(new ReactiveContract(new SpringMvcContract()))
                .target(CapsVisualizationClient.class, servicesProperties.getCapsVisualUrl());
    }

    @Bean
    public MachineLearningClient machineLearningClient() {
        return WebReactiveFeign.<MachineLearningClient>builder()
                .contract(new ReactiveContract(new SpringMvcContract()))
                .target(MachineLearningClient.class, servicesProperties.getMachineLearningUrl());
    }

}
