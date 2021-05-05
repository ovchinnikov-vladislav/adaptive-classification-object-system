package rsoi.lab3.microservices.front.client;


import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import rsoi.lab3.microservices.front.conf.FeignErrorDecoder;
import rsoi.lab3.microservices.front.model.AuthorizationCode;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

@FeignClient(name="session-service", configuration = FeignErrorDecoder.class)
public interface SessionClient {
    @PostMapping(value = "/api/v1/oauth/token", produces = MediaType.APPLICATION_JSON_UTF8_VALUE,
            consumes = MediaType.APPLICATION_JSON_UTF8_VALUE)
    Map<String, String> getToken(@RequestBody HashMap<String, String> requestDto);

    @GetMapping(value = "/api/v1/oauth")
    AuthorizationCode getCode(@RequestParam(value = "response_type") String responseType,
                              @RequestParam(value = "client_id") UUID clientId,
                              @RequestParam(value = "redirect_uri") String redirectUri,
                              @RequestHeader("Authorization") String token);

    @PostMapping(value = "/api/v1/oauth/token/validity")
    Map<String, String> validityToken(@RequestBody HashMap<String, String> token);
}
