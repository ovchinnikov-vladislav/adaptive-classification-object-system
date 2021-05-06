package bmstu.dynamic.simulator.client;

import feign.Headers;
import feign.Param;
import feign.RequestLine;
import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

@FeignClient(name = "auth-api", url = "${auth.client.url}")
public interface AuthApi {

    @RequestMapping(method = RequestMethod.GET, value = "/realms/bmstu/protocol/openid-connect/logout?redirect_uri=/")
    void logout();

}
