package ru.bmstu.adapt.config;

import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.http.HttpMethod;
import org.springframework.security.config.annotation.method.configuration.EnableReactiveMethodSecurity;
import org.springframework.security.config.annotation.web.reactive.EnableWebFluxSecurity;
import org.springframework.security.config.web.server.SecurityWebFiltersOrder;
import org.springframework.security.config.web.server.ServerHttpSecurity;
import org.springframework.security.oauth2.client.oidc.web.server.logout.OidcClientInitiatedServerLogoutSuccessHandler;
import org.springframework.security.oauth2.client.registration.ReactiveClientRegistrationRepository;
import org.springframework.security.web.server.SecurityWebFilterChain;
import org.springframework.security.web.server.authentication.logout.ServerLogoutSuccessHandler;
import org.springframework.security.web.server.ui.LoginPageGeneratingWebFilter;
import org.springframework.security.web.server.util.matcher.ServerWebExchangeMatchers;

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

    private final ServicesProperties servicesProperties;

    @Autowired
    ReactiveClientRegistrationRepository clientRegistrationRepository;

    ServerLogoutSuccessHandler oidcLogoutSuccessHandler() {
        var successHandler = new OidcClientInitiatedServerLogoutSuccessHandler(clientRegistrationRepository);
        successHandler.setPostLogoutRedirectUri(URI.create(servicesProperties.getExternalAuthUrl()));
        return successHandler;
    }


    @Bean
    SecurityWebFilterChain springSecurityFilterChain(ServerHttpSecurity http) {

        http
                .csrf().disable()
                .authorizeExchange()
                .pathMatchers("/error/**", "/login**", "/video**").permitAll()
                .pathMatchers("/css/**", "/img/**", "/js/**", "/scss/**", "/vendor/**").permitAll()
                .anyExchange().authenticated()
                .and().logout().logoutSuccessHandler(oidcLogoutSuccessHandler())
                // enable OAuth2/OIDC
                .and().oauth2Login()
                .and().oauth2ResourceServer().jwt().jwkSetUri(jwtUri)
                ;
        return http.build();
    }

}