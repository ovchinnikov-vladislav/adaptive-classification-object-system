package rsoi.lab3.microservices.front.exception;

public class ServiceAccessException extends RuntimeException {
    public ServiceAccessException(String message) {
        super(message);
    }
}
