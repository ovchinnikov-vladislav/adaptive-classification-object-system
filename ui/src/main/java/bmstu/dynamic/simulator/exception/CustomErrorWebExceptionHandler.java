package bmstu.dynamic.simulator.exception;

import org.springframework.boot.autoconfigure.web.ResourceProperties;
import org.springframework.boot.autoconfigure.web.reactive.error.AbstractErrorWebExceptionHandler;
import org.springframework.boot.web.reactive.error.ErrorAttributes;
import org.springframework.context.ApplicationContext;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.context.ReactiveSecurityContextHolder;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.net.URI;
import java.security.Principal;
import java.util.Objects;

import static org.springframework.web.reactive.function.server.RequestPredicates.all;
import static org.springframework.web.reactive.function.server.RouterFunctions.route;

public class CustomErrorWebExceptionHandler extends AbstractErrorWebExceptionHandler {

    public CustomErrorWebExceptionHandler(ErrorAttributes errorAttributes, ResourceProperties resourceProperties, ApplicationContext applicationContext) {
        super(errorAttributes, resourceProperties, applicationContext);
    }

    @Override
    protected RouterFunction<ServerResponse> getRoutingFunction(ErrorAttributes errorAttributes) {
        return route(all(), this::renderErrorResponse);
    }

    private Mono<ServerResponse> renderErrorResponse(ServerRequest serverRequest) {
        Throwable throwable = (Throwable) serverRequest
                .attribute("org.springframework.boot.web.reactive.error.DefaultErrorAttributes.ERROR")
                .orElseThrow(() -> new IllegalStateException("Missing exception attribute in ServerWebExchange"));

        String code = "";
        if (throwable != null && throwable.getMessage() != null) {
            code = throwable.getMessage().substring(0, 3);
        }

        String finalCode = code;
        return Mono.zip(
                serverRequest.principal(),
                serverRequest.bodyToMono(String.class)
        ).flatMap(tuple -> {
            Principal principal = tuple.getT1();
            if (principal.getName() != null) {
                return ServerResponse.temporaryRedirect(URI.create("/lk_error/" + finalCode)).build();
            } else {
                return ServerResponse.temporaryRedirect(URI.create("/error/" + finalCode)).build();
            }
        });

    }

}