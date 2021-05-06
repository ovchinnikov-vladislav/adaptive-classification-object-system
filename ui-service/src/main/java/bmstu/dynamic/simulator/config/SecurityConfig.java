package bmstu.dynamic.simulator.config;

import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.security.config.annotation.method.configuration.EnableReactiveMethodSecurity;
import org.springframework.security.config.annotation.web.reactive.EnableWebFluxSecurity;
import org.springframework.security.config.web.server.ServerHttpSecurity;
import org.springframework.security.oauth2.client.oidc.web.server.logout.OidcClientInitiatedServerLogoutSuccessHandler;
import org.springframework.security.oauth2.client.registration.ReactiveClientRegistrationRepository;
import org.springframework.security.web.server.SecurityWebFilterChain;
import org.springframework.security.web.server.authentication.logout.ServerLogoutSuccessHandler;

import java.net.URI;

/**
 * @author Rob Winch
 * @since 5.1
 */
@EnableWebFluxSecurity
@EnableReactiveMethodSecurity
@RequiredArgsConstructor
public class SecurityConfig {

    @Value("${spring.security.oauth2.resource-server.jwt.jwk-set-uri}")
    private String jwtUri;

    private final ReactiveClientRegistrationRepository clientRegistrationRepository;

    @Bean
    SecurityWebFilterChain springSecurityFilterChain(ServerHttpSecurity http) throws Exception {
        http
                .csrf().disable()
                .authorizeExchange()
                .pathMatchers("/error/**").permitAll()
                .pathMatchers("/css/**", "/img/**", "/js/**", "/scss/**", "/vendor/**").permitAll()
                .anyExchange().authenticated()
                .and()
                .oauth2Login()
                .and()
                .logout(logout -> logout.logoutSuccessHandler(oidcLogoutSuccessHandler()))
                .oauth2ResourceServer()
                .jwt()
                .jwkSetUri(jwtUri);
        return http.build();
    }

    private ServerLogoutSuccessHandler oidcLogoutSuccessHandler() {
        OidcClientInitiatedServerLogoutSuccessHandler oidcLogoutSuccessHandler =
                new OidcClientInitiatedServerLogoutSuccessHandler(clientRegistrationRepository);

        oidcLogoutSuccessHandler.setPostLogoutRedirectUri(URI.create("http://localhost:8080/"));

        return oidcLogoutSuccessHandler;
    }

}