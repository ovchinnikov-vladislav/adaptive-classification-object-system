package rsoi.lab3.microservices.front.controller;

import com.netflix.hystrix.exception.HystrixRuntimeException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.validation.BindException;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.servlet.config.annotation.EnableWebMvc;
import rsoi.lab3.microservices.front.entity.user.User;
import rsoi.lab3.microservices.front.exception.AuthenticationException;
import rsoi.lab3.microservices.front.exception.HttpCanNotCreateException;
import rsoi.lab3.microservices.front.exception.HttpNotFoundException;
import rsoi.lab3.microservices.front.exception.ServiceAccessException;
import rsoi.lab3.microservices.front.exception.feign.ClientAuthenticationExceptionWrapper;
import rsoi.lab3.microservices.front.exception.feign.ClientBadResponseExceptionWrapper;
import rsoi.lab3.microservices.front.exception.feign.ClientNotFoundExceptionWrapper;
import rsoi.lab3.microservices.front.exception.feign.OtherServiceExceptionWrapper;
import rsoi.lab3.microservices.front.model.ErrorResponse;

import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.util.Date;
import java.util.List;
import java.util.stream.Collectors;

@ControllerAdvice()
public class ExceptionController {

    private static final Logger logger = LoggerFactory.getLogger(ExceptionHandler.class);

    @ResponseStatus(HttpStatus.BAD_REQUEST)
    @ExceptionHandler({BindException.class, MethodArgumentNotValidException.class})
    @ResponseBody
    public ErrorResponse bindExceptionOrMethodArgumentNotValidExceptionHandler(Exception exc) {
        List<FieldError> errors = null;
        if (exc instanceof BindException)
            errors = ((BindException) exc).getBindingResult().getFieldErrors();
        else if (exc instanceof MethodArgumentNotValidException)
            errors = ((MethodArgumentNotValidException) exc).getBindingResult().getFieldErrors();
        if (errors != null) {
            String result = errors.stream().map(err -> String.format("%s: [%s]", err.getField(), err.getDefaultMessage()))
                    .collect(Collectors.joining("; "));
            logger.error("Bad Request: {}", result);
            return new ErrorResponse(HttpStatus.BAD_REQUEST.value(), HttpStatus.BAD_REQUEST.getReasonPhrase(), result);
        }
        return new ErrorResponse(HttpStatus.BAD_REQUEST.value(), HttpStatus.BAD_REQUEST.getReasonPhrase(), exc.getMessage());
    }

    @ResponseStatus(HttpStatus.BAD_REQUEST)
    @ExceptionHandler(ClientBadResponseExceptionWrapper.class)
    @ResponseBody
    public ErrorResponse httpCanNotCreateExceptionHandler(ClientBadResponseExceptionWrapper exc) {
        logger.error("Bad Request: {}", exc.getMessage());
        return new ErrorResponse(HttpStatus.BAD_REQUEST.value(), HttpStatus.BAD_REQUEST.getReasonPhrase(), exc.getMessage());
    }

    @ResponseStatus(HttpStatus.BAD_REQUEST)
    @ExceptionHandler(ClientNotFoundExceptionWrapper.class)
    @ResponseBody
    public ErrorResponse httpNotFoundExceptionHandler(ClientNotFoundExceptionWrapper exc) {
        logger.error("Bad Request: {}", exc.getMessage());
        return new ErrorResponse(HttpStatus.BAD_REQUEST.value(), HttpStatus.BAD_REQUEST.getReasonPhrase(), exc.getMessage());
    }

    @ResponseStatus(HttpStatus.UNAUTHORIZED)
    @ExceptionHandler(ClientAuthenticationExceptionWrapper.class)
    @ResponseBody
    public ErrorResponse authenticationExceptionHandler(ClientAuthenticationExceptionWrapper exc) {
        logger.error("Bad Request: {}", exc.getMessage());
        return new ErrorResponse(HttpStatus.UNAUTHORIZED.value(), HttpStatus.UNAUTHORIZED.getReasonPhrase(), exc.getMessage());
    }

    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
    @ExceptionHandler(HttpClientErrorException.class)
    @ResponseBody
    public ErrorResponse httpClientErrorExceptionHandler(HttpClientErrorException exc) {
        logger.error("Interval Server Error: {}", exc.getMessage());
        return new ErrorResponse(HttpStatus.INTERNAL_SERVER_ERROR.value(), HttpStatus.INTERNAL_SERVER_ERROR.getReasonPhrase(), exc.getMessage());
    }

    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
    @ExceptionHandler(OtherServiceExceptionWrapper.class)
    @ResponseBody
    public ErrorResponse serviceAccessExceptionHandler(OtherServiceExceptionWrapper exc) {
        logger.error("Interval Server Error: {}", exc.getMessage());
        return new ErrorResponse(HttpStatus.INTERNAL_SERVER_ERROR.value(), HttpStatus.INTERNAL_SERVER_ERROR.getReasonPhrase(), exc.getMessage());
    }

}
