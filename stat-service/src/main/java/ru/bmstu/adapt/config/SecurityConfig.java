package ru.bmstu.adapt.config;

//import lombok.RequiredArgsConstructor;
//import org.springframework.beans.factory.annotation.Value;
//import org.springframework.context.annotation.Configuration;
//import org.springframework.security.config.annotation.web.builders.HttpSecurity;
//import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
//import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;
//import org.springframework.security.oauth2.client.registration.ClientRegistrationRepository;
//import org.springframework.security.web.authentication.logout.LogoutSuccessHandler;
//import org.springframework.security.oauth2.client.oidc.web.logout.OidcClientInitiatedLogoutSuccessHandler;
//
//import java.net.URI;

//@Configuration
//@EnableWebSecurity
//@RequiredArgsConstructor
//public class SecurityConfig extends WebSecurityConfigurerAdapter {
//
//    @Value("${spring.security.oauth2.resource-server.jwt.jwk-set-uri}")
//    private String jwtUri;
//
//    private final ClientRegistrationRepository clientRegistrationRepository;
//
//    @Override
//    protected void configure(HttpSecurity http) throws Exception {
//        http
//                .csrf().disable()
//                .authorizeRequests()
//                .antMatchers("/error/**").permitAll()
//                .anyRequest().authenticated()
//                .and()
//                .oauth2Login()
//                .and()
//                .logout(logout -> logout.logoutSuccessHandler(oidcLogoutSuccessHandler()))
//                .oauth2ResourceServer()
//                .jwt()
//                .jwkSetUri(jwtUri);
//        http.build();
//    }
//
//    private LogoutSuccessHandler oidcLogoutSuccessHandler() {
//        var oidcLogoutSuccessHandler = new OidcClientInitiatedLogoutSuccessHandler(clientRegistrationRepository);
//
//        oidcLogoutSuccessHandler.setPostLogoutRedirectUri(URI.create("http://localhost:8080/"));
//
//        return oidcLogoutSuccessHandler;
//    }
//
//}