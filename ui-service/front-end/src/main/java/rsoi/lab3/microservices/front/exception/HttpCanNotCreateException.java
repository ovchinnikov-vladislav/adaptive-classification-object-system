package rsoi.lab3.microservices.front.exception;

public class HttpCanNotCreateException extends RuntimeException {
    public HttpCanNotCreateException(String message) {
        super(message);
    }
}
