package bmstu.dynamic.simulator.exception;

import bmstu.dynamic.simulator.exception.impl.NotFoundWebException;
import bmstu.dynamic.simulator.service.ModelService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@Slf4j
@RestControllerAdvice
@RequiredArgsConstructor
public class GlobalExceptionHandler {

    private final ModelService modelService;

    @ExceptionHandler(NotFoundWebException.class)
    @ResponseStatus(HttpStatus.NOT_FOUND)
    public String handleResourceNotFoundException(Model model) {
        modelService.prepareStandardModelForTemplate(model);

        return "404";
    }

}
