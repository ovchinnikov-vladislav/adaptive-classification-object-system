package bmstu.dynamic.simulator.service;

import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.ui.Model;

import java.time.LocalDate;

@Service
public class ModelService {

    public void prepareStandardModelForTemplate(Model model) {
        model.addAttribute("now_year", LocalDate.now().getYear());
        model.addAttribute("copyright", "Vladislav Ovchinnikov");
    }

    public void prepareErrorModelForTemplate(Model model, Integer status) {
        model.addAttribute("status", "Undefined");

        if (HttpStatus.valueOf(status) == HttpStatus.NOT_FOUND) {
            model.addAttribute("status", status);
            model.addAttribute("text", "Page Not Found");
            model.addAttribute("additional_text",  "It looks like you found a glitch in the matrix...");
        } else if (HttpStatus.valueOf(status) == HttpStatus.BAD_REQUEST) {
            model.addAttribute("status", status);
            model.addAttribute("text", "Page Not Found");
            model.addAttribute("additional_text",  "It looks like you found a glitch in the matrix...");
        } else if (HttpStatus.valueOf(status).is5xxServerError()) {
            model.addAttribute("status", status);
            model.addAttribute("text", "Fatal error");
        }

        model.addAttribute("now_year", LocalDate.now().getYear());
        model.addAttribute("copyright", "Vladislav Ovchinnikov");
    }
}
