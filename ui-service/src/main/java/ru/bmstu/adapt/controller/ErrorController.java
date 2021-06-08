package ru.bmstu.adapt.controller;

import ru.bmstu.adapt.service.ModelService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.server.reactive.ServerHttpResponse;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
@RequestMapping("/")
@RequiredArgsConstructor
public class ErrorController {

    private final ModelService modelService;

    @GetMapping("/error/{status}")
    public String error(@PathVariable Integer status, Model model, ServerHttpResponse response) {
        modelService.prepareErrorModelForTemplate(model, status, response);
        return "pages/error";
    }

    @GetMapping("/error_content/{status}")
    public String errorContent(@PathVariable Integer status, Model model, ServerHttpResponse response) {
        modelService.prepareErrorModelForTemplate(model, status, response);
        return "fragments/error :: error_content";
    }

}
