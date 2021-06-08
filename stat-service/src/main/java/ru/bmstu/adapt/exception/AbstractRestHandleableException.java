package ru.bmstu.adapt.exception;

import lombok.Getter;
import org.springframework.http.HttpStatus;

@Getter
public abstract class AbstractRestHandleableException extends RuntimeException {
    private static final long serialVersionUID = -3416823984750319182L;

    private final HttpStatus httpStatus;

    public AbstractRestHandleableException(String message, HttpStatus httpStatus) {
        super(message);
        this.httpStatus = httpStatus;
    }

    public AbstractRestHandleableException(String message, Throwable throwable, HttpStatus httpStatus) {
        super(message, throwable);
        this.httpStatus = httpStatus;
    }
}