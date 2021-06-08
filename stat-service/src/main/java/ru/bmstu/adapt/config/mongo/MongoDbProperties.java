package ru.bmstu.adapt.config.mongo;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

import java.util.List;

@Data
@Configuration
@ConfigurationProperties("mongo")
public class MongoDbProperties {

    private List<String> hosts;
    private boolean ssl;
    private String database;
    private String username;
    private String password;
    private Integer port;

}
