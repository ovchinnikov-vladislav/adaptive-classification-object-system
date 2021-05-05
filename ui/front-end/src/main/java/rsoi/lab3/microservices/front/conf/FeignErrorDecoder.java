package rsoi.lab3.microservices.front.conf;

import com.fasterxml.jackson.databind.ObjectMapper;
import feign.Response;
import feign.codec.ErrorDecoder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import rsoi.lab3.microservices.front.exception.AuthenticationException;
import rsoi.lab3.microservices.front.exception.HttpCanNotCreateException;
import rsoi.lab3.microservices.front.exception.HttpNotFoundException;
import rsoi.lab3.microservices.front.exception.ServiceAccessException;
import rsoi.lab3.microservices.front.exception.feign.ClientAuthenticationExceptionWrapper;
import rsoi.lab3.microservices.front.exception.feign.ClientBadResponseExceptionWrapper;
import rsoi.lab3.microservices.front.exception.feign.ClientNotFoundExceptionWrapper;
import rsoi.lab3.microservices.front.exception.feign.OtherServiceExceptionWrapper;
import rsoi.lab3.microservices.front.model.ErrorResponse;

import java.io.IOException;

@Component
public class FeignErrorDecoder implements ErrorDecoder {
    Logger logger = LoggerFactory.getLogger(this.getClass());

    @Override
    public Exception decode(String methodKey, Response response) {
        ErrorResponse errorResponse = new ErrorResponse();
        errorResponse.setMessage("Undefined Error");
        errorResponse.setStatus(response.status());
        try {
            String message = new String(response.body().asInputStream().readAllBytes());
            errorResponse = new ObjectMapper().readValue(message, ErrorResponse.class);
            logger.info("Status code " + response.status() + ", methodKey = " + methodKey);
            logger.info("Message: " + errorResponse.getMessage());
        } catch (Exception exc) {
            logger.error(exc.getMessage());
        }
        switch (response.status()) {
            case 400:
                return new ClientBadResponseExceptionWrapper(errorResponse.getMessage());
            case 404:
                return new ClientNotFoundExceptionWrapper(errorResponse.getMessage());
            case 401:
                return new ClientAuthenticationExceptionWrapper(errorResponse.getMessage());
            case 500:
                return new OtherServiceExceptionWrapper(errorResponse.getMessage());
        }

        return new Exception(errorResponse.getMessage());
    }
}
