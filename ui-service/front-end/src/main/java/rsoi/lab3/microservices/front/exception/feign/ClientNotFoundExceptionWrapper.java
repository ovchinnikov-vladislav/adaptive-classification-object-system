package rsoi.lab3.microservices.front.exception.feign;

import com.netflix.hystrix.exception.HystrixBadRequestException;

public class ClientNotFoundExceptionWrapper extends HystrixBadRequestException {
    public ClientNotFoundExceptionWrapper(String message) {
        super(message);
    }
}
