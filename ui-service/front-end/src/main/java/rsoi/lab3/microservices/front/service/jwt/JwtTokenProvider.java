package rsoi.lab3.microservices.front.service.jwt;

import io.jsonwebtoken.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Component;
import rsoi.lab3.microservices.front.client.GatewayClient;
import rsoi.lab3.microservices.front.client.SessionClient;
import rsoi.lab3.microservices.front.entity.user.Role;
import rsoi.lab3.microservices.front.entity.user.User;

import javax.servlet.http.Cookie;
import java.util.*;

@Component
public class JwtTokenProvider {

    private static final Logger logger = LoggerFactory.getLogger(JwtTokenProvider.class);

    private String secretKey = "d282dc035756736e54761761cc52bef78e3c473fa7de8f617c14f0e0ae7044aae8ba4b7bed7d532d4af91122e50b39a8bb99e320f72094547d7cae108e928460";

    @Autowired
    private GatewayClient gatewayClient;
    @Autowired
    private SessionClient sessionClient;

    @Bean
    public BCryptPasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    public String resolveToken(Cookie cookie) {
        return cookie.getValue();
    }

    public Map<String, String> generationToken(HashMap<String, String> map) {
        return sessionClient.getToken(map);
    }

    public UUID getUserIdFromToken(String token) {
        Claims claims = getJwsClaimsFromToken(token).getBody();
        return UUID.fromString(claims.get("user_id", String.class));
    }

    public String getUsernameFromToken(String token) {
        Claims claims = getJwsClaimsFromToken(token).getBody();
        return claims.get("username", String.class);
    }

    public User getUserByToken(String token) {
        User user = new User();
        Claims claims = getJwsClaimsFromToken(token).getBody();
        user.setId(UUID.fromString(claims.get("user_id", String.class)));
        user.setUsername(claims.get("username", String.class));
        user.setRoles((List<Role>) claims.get("roles"));
        return user;
    }

    public boolean validateAccessToken(String token) {
        try {
            Jws<Claims> claims = getJwsClaimsFromToken(token);
            Date date = claims.getBody().getExpiration();
            if (date.before(new Date())) {
                logger.warn("JWT Token is expired.");
                return false;
            }
            return true;
        } catch (JwtException | IllegalArgumentException exc) {
            if (exc instanceof ExpiredJwtException) {
                logger.warn("JWT Token is expired.");
            } else {
                logger.warn("JWT Token is invalid.");
            }
            return false;
        }
    }

    private Jws<Claims> getJwsClaimsFromToken(String token) {
        return Jwts.parser().setSigningKey(secretKey).parseClaimsJws(token);
    }

}
