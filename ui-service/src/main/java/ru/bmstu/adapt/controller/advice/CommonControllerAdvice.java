package bmstu.dynamic.simulator.controller.advice;

import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ModelAttribute;

import java.time.LocalDate;
import java.util.Objects;

@ControllerAdvice
public class CommonControllerAdvice {

    @ModelAttribute
    public void copyrightData(Model model) {
        model.addAttribute("now_year", LocalDate.now().getYear());
        model.addAttribute("copyright", "Владислав Овчинников");
    }

}
