package rsoi.lab3.microservices.front.service;

import org.apache.commons.codec.digest.DigestUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import rsoi.lab3.microservices.front.client.GatewayClient;
import rsoi.lab3.microservices.front.client.SessionClient;
import rsoi.lab3.microservices.front.entity.user.User;
import rsoi.lab3.microservices.front.model.AuthorizationCode;
import rsoi.lab3.microservices.front.model.OAuthClient;

import java.util.UUID;

@Service
public class AuthService {

    @Autowired
    private GatewayClient gatewayClient;
    @Autowired
    private SessionClient sessionClient;

    public User create(User user) {
        //String md5Hex = DigestUtils.md5Hex(user.getPassword()).toUpperCase();
        //user.setPassword(md5Hex);
        return gatewayClient.createUser(user).orElse(null);
    }

    public OAuthClient findClientById(UUID id, String token) {
        return gatewayClient.findClientById(id, "Bearer " + token);
    }

    public AuthorizationCode getCode(UUID clientId, String redirectUri, String token) {
        return sessionClient.getCode("code", clientId, redirectUri, "Bearer " + token);
    }
}
