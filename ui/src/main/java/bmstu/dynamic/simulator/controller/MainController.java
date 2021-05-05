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

    @GetMapping("/blank")
    public String blank(Model model) {
        modelService.prepareStandardModelForTemplate(model);

        return "blank";
    }

    @GetMapping("/buttons")
    public String buttons(Model model) {
        modelService.prepareStandardModelForTemplate(model);

        return "buttons";
    }

    @GetMapping("/cards")
    public String cards(Model model) {
        modelService.prepareStandardModelForTemplate(model);

        return "cards";
    }

    @GetMapping("/charts")
    public String charts(Model model) {
        modelService.prepareStandardModelForTemplate(model);

        return "charts";
    }

    @GetMapping("/tables")
    public String tables(Model model) {
        modelService.prepareStandardModelForTemplate(model);

        return "tables";
    }

    @GetMapping("/utilities-animation")
    public String utilitiesAnimation(Model model) {
        modelService.prepareStandardModelForTemplate(model);

        return "utilities-animation";
    }

    @GetMapping("/utilities-border")
    public String utilitiesBorder(Model model) {
        modelService.prepareStandardModelForTemplate(model);

        return "utilities-border";
    }

    @GetMapping("/utilities-color")
    public String utilitiesColor(Model model) {
        modelService.prepareStandardModelForTemplate(model);

        return "utilities-color";
    }

    @GetMapping("/utilities-other")
    public String utilitiesOther(Model model) {
        modelService.prepareStandardModelForTemplate(model);

        return "utilities-other";
    }

    @GetMapping("/error/{status}")
    public String error(Model model, @PathVariable Integer status) {
        modelService.prepareErrorModelForTemplate(model, status);

        return "error";
    }

}
