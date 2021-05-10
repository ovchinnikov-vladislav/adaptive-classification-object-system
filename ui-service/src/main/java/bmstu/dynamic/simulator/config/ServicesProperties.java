package bmstu.dynamic.simulator.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Data
@Configuration
@ConfigurationProperties("services")
public class ServicesProperties {

    private String authUrl;
    private String capsVisualUrl;
    private String machineLearningUrl;
    private String statUrl;

}