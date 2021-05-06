package rsoi.lab3.microservices.front.exception;

public class HttpNotFoundException extends RuntimeException {
    public HttpNotFoundException(String message) {
        super(message);
    }
}
