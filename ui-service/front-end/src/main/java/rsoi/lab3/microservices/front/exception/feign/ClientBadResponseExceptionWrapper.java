package rsoi.lab3.microservices.front.exception.feign;

import com.netflix.hystrix.exception.HystrixBadRequestException;

public class ClientBadResponseExceptionWrapper extends HystrixBadRequestException {
    public ClientBadResponseExceptionWrapper(String message) {
        super(message);
    }
}
