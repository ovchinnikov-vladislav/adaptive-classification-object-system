/*
 * This Java source file was generated by the Gradle 'init' task.
 */
package rsoi.lab3.microservices.front;

import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.netflix.eureka.EnableEurekaClient;
import org.springframework.cloud.openfeign.EnableFeignClients;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.PropertySource;

@EnableEurekaClient
@EnableFeignClients
@SpringBootApplication(scanBasePackages = "rsoi.lab3.microservices.front.*")
@PropertySource("classpath:application.properties")
public class App {

    public static void main(String[] args) {
        SpringApplication.run(App.class, args);
    }

    @Bean
    public CommandLineRunner commandLineRunner(ApplicationContext ctx) {
        return str -> System.out.println("Frontend");
    }
}
