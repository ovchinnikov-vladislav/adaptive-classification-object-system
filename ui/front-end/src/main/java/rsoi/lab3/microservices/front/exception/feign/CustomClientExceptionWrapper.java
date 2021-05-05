package rsoi.lab3.microservices.front.exception.feign;

import com.netflix.hystrix.exception.HystrixBadRequestException;

public class CustomClientExceptionWrapper extends HystrixBadRequestException {
    public CustomClientExceptionWrapper(String message) {
        super(message);
    }
}
