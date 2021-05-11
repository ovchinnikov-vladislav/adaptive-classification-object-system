package bmstu.dynamic.simulator.config;


import bmstu.dynamic.simulator.config.mongo.MongoDbProperties;
import com.mongodb.MongoClientSettings;
import com.mongodb.MongoCredential;
import com.mongodb.ServerAddress;
import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.mongodb.core.MongoTemplate;

import java.util.ArrayList;
import java.util.List;


@Configuration
public class StatConfig {

    @Bean
    public MongoClient mongoClient(MongoDbProperties mongoDbProperties) {

        List<ServerAddress> serverAddresses = new ArrayList<>();
        mongoDbProperties.getHosts().forEach(
                (host) -> serverAddresses.add(new ServerAddress(host, mongoDbProperties.getPort() == null ? 27018 :
                        mongoDbProperties.getPort()))
        );

        return MongoClients.create(
                MongoClientSettings.builder()
                        .applyToClusterSettings(builder ->
                                builder.hosts(serverAddresses))
                        .applyToSslSettings(builder ->
                                builder.enabled(mongoDbProperties.isSsl()))
                        .credential(MongoCredential
                                .createCredential(mongoDbProperties.getUsername(),
                                        mongoDbProperties.getDatabase(),
                                        mongoDbProperties.getPassword()
                                                .toCharArray()))
                        .build());

    }

    @Bean
    public MongoTemplate mongoTemplate(MongoDbProperties mongoDbProperties) {
        return new MongoTemplate(mongoClient(mongoDbProperties), mongoDbProperties.getDatabase());
    }

}
