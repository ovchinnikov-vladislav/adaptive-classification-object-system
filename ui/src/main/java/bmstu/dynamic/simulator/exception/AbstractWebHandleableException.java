package bmstu.dynamic.simulator.exception;

import org.springframework.http.HttpStatus;

public abstract class AbstractWebHandleableException extends RuntimeException {

    private static final long serialVersionUID = -3416823984750319182L;

    private final HttpStatus httpStatus;

    public AbstractWebHandleableException(String message, HttpStatus httpStatus) {
        super(message);
        this.httpStatus = httpStatus;
    }

    public AbstractWebHandleableException(String message, Throwable throwable, HttpStatus httpStatus) {
        super(message, throwable);
        this.httpStatus = httpStatus;
    }

}
