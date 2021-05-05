package bmstu.dynamic.simulator.controller;

import bmstu.dynamic.simulator.service.ModelService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.reactive.function.server.ServerRequest;

@Controller
@RequestMapping("/")
@RequiredArgsConstructor
public class MainController {

    private final ModelService modelService;

    @GetMapping("/")
    public String index(Model model) {
        modelService.prepareStandardModelForTemplate(model);

        return "index";
    }

    @GetMapping("/error/{status}")
    public String error(Model model, @PathVariable Integer status) {
        modelService.prepareErrorModelForTemplate(model, status);

        return "error";
    }

}
