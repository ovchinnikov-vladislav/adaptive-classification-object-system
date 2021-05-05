package bmstu.dynamic.simulator.exception;

import org.springframework.boot.autoconfigure.web.ResourceProperties;
import org.springframework.boot.autoconfigure.web.reactive.error.AbstractErrorWebExceptionHandler;
import org.springframework.boot.web.reactive.error.ErrorAttributes;
import org.springframework.context.ApplicationContext;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.net.URI;

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

        // TODO: fix get status code error webflux
        String code = throwable.getMessage().substring(0, 3);

        return ServerResponse.temporaryRedirect(URI.create("/error/" + code)).build();
    }

}