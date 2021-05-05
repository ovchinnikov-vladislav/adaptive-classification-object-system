package rsoi.lab3.microservices.front.exception.feign;

import com.netflix.hystrix.exception.HystrixBadRequestException;

public class OtherServiceExceptionWrapper extends HystrixBadRequestException {
    public OtherServiceExceptionWrapper(String message) {
        super(message);
    }
}
