package ru.bmstu.adapt.exception.impl;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ResponseStatus;

@ResponseStatus(HttpStatus.NOT_FOUND)
public class VideoNotFoundException extends NotFoundWebException {

    public VideoNotFoundException() {
        super("video was not found");
    }
}