package rsoi.lab3.microservices.front.exception.feign;

import com.netflix.hystrix.exception.HystrixBadRequestException;

public class ClientAuthenticationExceptionWrapper extends HystrixBadRequestException {
    public ClientAuthenticationExceptionWrapper(String message) {
        super(message);
    }

    public ClientAuthenticationExceptionWrapper(String message, Throwable cause) {
        super(message, cause);
    }
}
