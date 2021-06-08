package ru.bmstu.adapt.config;

import ru.bmstu.adapt.client.CapsVisualizationClient;
import ru.bmstu.adapt.client.MachineLearningClient;
import lombok.RequiredArgsConstructor;
import org.springframework.cloud.openfeign.support.SpringMvcContract;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
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
