package ru.bmstu.adapt.exception.impl;

import ru.bmstu.adapt.exception.AbstractWebHandleableException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ResponseStatus;

/**
 * Exception thrown when the video location is inaccessible
 * or does not exist.
 */
@Slf4j
@ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
public class BadResourceLocationException extends AbstractWebHandleableException {

    public BadResourceLocationException(String message) {
        super(message, HttpStatus.BAD_REQUEST);
    }

    public BadResourceLocationException(String message, Throwable throwable) {
        super(message, throwable, HttpStatus.BAD_REQUEST);
    }

}